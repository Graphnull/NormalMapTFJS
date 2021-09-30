//import { env } from '@tensorflow/tfjs-core';
//import { getGlslDifferences } from '@tensorflow/tfjs-backend-webgl/dist/glsl_version';
//import { getKernel } from '@tensorflow/tfjs-core/dist/kernel_registry';
let tf = require('@tensorflow/tfjs-node');
//import '@tensorflow/'
let fs = require('fs')
let { opticalFlowFind, blur } = require('nodleten/src/kernels/opticatFlowKernels')
let custBlur = (fld, ks, step) => {
  fld = tf.conv2d(
    tf.stack(fld.unstack(-1).map(v => v.reshape([v.shape[0], v.shape[1], 1]))),
    tf.fill([ks, ks, 1, 1], 1 / (ks * ks)), [step, step], 'valid')
  fld = tf.concat(fld.unstack(), -1)
  return fld;
}
let minIndex = (arr) => {
  let i = 0;
  let minV = Infinity;
  for (let j = 0; j < arr.length; j++) {
    if (minV > arr[j]) {
      minV = arr[j];
      i = j
    }
  }
  return i;
}
//import '@tensorflow/tfjs-backend-webgl/dist/register_all_kernels.js';
//const glsl = getGlslDifferences();
let ks = 7
let hks = Math.floor(ks / 2)
let dil = 1;
let dk = ks / dil

let step = 4
let height = 436 / 4
let width = 1024 / 4

let run = async () => {
  //console.log('tf', tf.engine().registryFactory, tf.getBackend());
  let log = console.log
  log('started')
  try {

    let video = tf.node.decodePng(fs.readFileSync('./public/frame_0002.png'), 3) //await new Promise((res) => { let img = new Image(); img.onload = () => res(img); img.src = '/NormalMapTFJS/frame_0002.png' })
    let video2 = tf.node.decodePng(fs.readFileSync('./public/frame_0001.png'), 3) //await new Promise((res) => { let img = new Image(); img.onload = () => res(img); img.src = '/NormalMapTFJS/frame_0001.png' })

    //let canvas = document.getElementById('result')
    //let canvas2 = document.getElementById('result2')

    await tf.setBackend('tensorflow');
    await new Promise((res) => setTimeout(res, 100))
    let height = video.shape[0];
    let width = video.shape[1];
    let flowData = tf.tensor(new Float32Array(fs.readFileSync('./public/frame_0001.flo').buffer.slice(3 * 4)), [height, width, 2])
    flowData = tf.concat([flowData, tf.zeros([height, width, 1])], -1)
    console.log('height: ', height, width);
    //let current = tf.zeros([1, height, width, 3]);
    //let pred = tf.zeros([1, height, width, 3]);
    let current2 = tf.zeros([1, height / 2, width / 2, 3]);
    let pred2 = tf.zeros([1, height / 2, width / 2, 3]);
    let current4 = tf.zeros([1, height / 4, width / 4, 3]);
    let pred4 = tf.zeros([1, height / 4, width / 4, 3]);

    let flowData2 = flowData.resizeBilinear([height / 2, width / 2]);

    let flowData4 = flowData.resizeBilinear([height / 4, width / 4]);
    let backend = tf.backend()
    let uv = tf.tensor([0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0,
      0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1,
      0, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2,
      0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3,
      0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4,
      0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5,
      0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 6, 6, 6], [ks * ks, 2]).sub(hks)

    let update = () => {
      let result = null;
      let time = new Date()
      try {

        // if (pred) {
        //   pred.dispose();
        // }
        // pred = current;

        current2 = video.expandDims().div(255)

        let temp = current2;
        current2 = current2.resizeBilinear([height / 2, width / 2])
        temp.dispose();
        temp = current4
        current4 = current2.resizeBilinear([height / 4, width / 4])
        temp.dispose();

        pred2 = video2.expandDims().div(255)
        temp = pred2;
        pred2 = pred2.resizeBilinear([height / 2, width / 2])
        temp.dispose();
        temp = pred4;
        pred4 = pred2.resizeBilinear([height / 4, width / 4])
        temp.dispose();

        result = tf.tidy(() => {
          let time = new Date()
          let x = 0;
          let y = 0;


          let t;
          let result = opticalFlowFind([tf.tensor([0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2]), pred4, current4], { step: step, kernelSize: ks });
          let px = result.shape[1]
          let py = result.shape[0]

          let shift = result.reshape([py * px, ks * ks * 1]).mean(-2);
          //tf.node.encodePng(shift.reshape([ks, ks, 1]).sub(0.01).mul(300).toInt()).then((data) => fs.writeFileSync('./resultshift.png', data));

          let shiftRes = minIndex(shift.dataSync());
          x = Math.floor(shiftRes % ks) - hks;
          y = Math.floor(shiftRes / ks) - hks;
          result = opticalFlowFind([tf.tensor([x, y, x, y, x, y, x, y], [2, 2, 2]), pred4, current4], { step: step, kernelSize: ks });
          result = custBlur(result.reshape([py, px, ks * ks]), 3, 2);
          result = result.reshape([result.shape[0], result.shape[1], ks * ks, 1]);

          let pos = result.argMin(-2);

          let xp = pos.mod(ks).sub(hks).add(x);
          let yp = pos.floorDiv(ks).sub(hks).add(y);

          let out = tf.concat([xp, yp, tf.zeros(xp.shape)], -1).mul(27).add(127).maximum(0).minimum(255).toInt();
          let fld = flowData4.slice([3, 3, 0], [Math.floor((flowData4.shape[0] - ks - hks) / step) * step + hks, Math.floor((flowData4.shape[1] - ks - hks) / step) * step + hks, 3]);
          fld = custBlur(fld, ks, step)
          fld = custBlur(fld, 3, 2)
          tf.node.encodePng(fld.mul(27).mul(0.5).add(127).maximum(0).minimum(255).toInt()).then((data) => fs.writeFileSync('./result1f.png', data));
          tf.node.encodePng(out).then((data) => fs.writeFileSync('./result1.png', data));

          // 2 step
          let back2 = tf.concat([xp, yp], -1)

          console.log('22', pred2.slice([0, hks, hks, 0], [1, Math.floor((pred2.shape[1] - ks - hks) / step) * step - hks - hks, Math.floor((pred2.shape[2] - ks - hks) / step) * step - hks - hks, 2]));
          result = opticalFlowFind([back2, pred2.slice([0, hks, hks, 0], [1, Math.floor((pred2.shape[1] - ks - hks) / step) * step - 6, Math.floor((pred2.shape[2] - ks - hks) / step) * step - 6, 2]),
            current2.slice([0, hks, hks, 0], [1, Math.floor((current2.shape[1] - ks - hks) / step) * step - 6, Math.floor((current2.shape[2] - ks - hks) / step) * step - 6, 2])], { step: step, kernelSize: ks });
          px = result.shape[1];
          py = result.shape[0];

          console.log(11);
          result = custBlur(result.reshape([py, px, ks * ks]), 3, 2);
          console.log('back2: ', back2, result);

          pos = result.argMin(-1)

          xp = pos.mod(dk).sub(hks).expandDims(-1);
          yp = pos.floorDiv(dk).sub(hks).expandDims(-1);

          out = tf.concat([xp, yp], -1)//.add(back2.resizeBilinear(xp.shape.slice(0, 2)))
            .concat([tf.zeros(xp.shape)], -1).mul(27).add(127).maximum(0).minimum(255).toInt();

          tf.node.encodePng(out).then((data) => fs.writeFileSync('./result2.png', data))

          fld = flowData2.slice([hks, hks, 0], [Math.floor((flowData2.shape[0] - ks - hks) / step) * step + hks, Math.floor((flowData2.shape[1] - ks - hks) / step) * step + hks, 3]);

          fld = fld.slice([hks, hks, 0], [Math.floor((fld.shape[0] - ks - hks) / step) * step + hks, Math.floor((fld.shape[1] - ks - hks) / step) * step + hks, 3]);
          fld = custBlur(fld, ks, step)
          fld = custBlur(fld, 3, 2)
          tf.node.encodePng(fld.mul(27).mul(0.5).add(127).maximum(0).minimum(255).toInt()).then((data) => fs.writeFileSync('./result2f.png', data));

          // 3 step

          let current = video.expandDims().div(255)
          let pred = video2.expandDims().div(255)
          let back3 = tf.concat([xp, yp], -1)
          result = opticalFlowFind([back3, pred.slice([0, hks, hks, 0], [1, Math.floor((pred.shape[1] - ks - hks) / step) * step - 6, Math.floor((pred.shape[2] - ks - hks) / step) * step - 6, 2]),
            current.slice([0, hks, hks, 0], [1, Math.floor((current.shape[1] - ks - hks) / step) * step - 6, Math.floor((current.shape[2] - ks - hks) / step) * step - 6, 2])], { step: step, kernelSize: ks });
          px = result.shape[1];
          py = result.shape[0];

          console.log(11);
          result = custBlur(result.reshape([py, px, ks * ks]), 3, 2);

          pos = result.reshape([result.shape[0], result.shape[1], ks * ks]).argMin(-1)

          xp = pos.mod(dk).sub(hks).expandDims(-1);
          yp = pos.floorDiv(dk).sub(hks).expandDims(-1);

          out = tf.concat([xp, yp], -1)//.add(back3.resizeBilinear(xp.shape.slice(0, 2)))
            .concat([tf.zeros(xp.shape)], -1).mul(27).add(127).maximum(0).minimum(255).toInt();

          tf.node.encodePng(out).then((data) => fs.writeFileSync('./result3.png', data))

          fld = flowData.slice([3, 3, 0], [Math.floor((flowData.shape[0] - ks - hks) / step) * step + hks, Math.floor((flowData.shape[1] - ks - hks) / step) * step + hks, 3]);
          fld = custBlur(fld, ks, step)
          fld = custBlur(fld, 3, 2)
          tf.node.encodePng(fld.mul(27).add(127).maximum(0).minimum(255).toInt()).then((data) => fs.writeFileSync('./result3f.png', data));
          //tf.browser.toPixels(out, canvas2)
          //console.log('out: ', out);
          console.log('time', new Date() - time)
          return out;
          let res = t//modelOpticalFlow.predict(t)
          let sum = res.reshape([res.shape[1], res.shape[2], dk * dk, 1]).relu().pow(2).sum(-2)

          let p = res.reshape([res.shape[1], res.shape[2], dk, dk, 1])

          p = p.concat([p], -1).pow(2).mul(uv).sum(-2).sum(-2)

          sum = sum.concat([sum], -1)
          let result2 = tf.concat([p.div(sum).add([y, x]), tf.zeros(p.shape)], -1).mul(3)
          return result2.mul(9).add(127).maximum(0).minimum(255).toInt();

        })
      } catch (err) {
        console.log('err: ', err);

        log(err.message)
        log(err.stack)
      }
      // tf.browser.toPixels(result, canvas).then(() => {
      //   ctime = ctime * 0.9 + (new Date() - time) * 0.1
      //   console.log('time', ctime)
      //   //log('req3')
      //   result.dispose();
      //   requestAnimationFrame(update)
      // }).catch(err => {
      //   console.log('err: ', err);
      //   log(err.message)
      //   log(err.stack)
      // })
    }
    update();
    //requestAnimationFrame(update)
  } catch (err) {
    console.log('err: ', err);
    log(err.message)
    log(err.stack)
  }
}


run()