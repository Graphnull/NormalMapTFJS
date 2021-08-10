import React from 'react'
import logo from './logo.svg';
import './App.css';
import { env } from '@tensorflow/tfjs-core';
import { getGlslDifferences } from '@tensorflow/tfjs-backend-webgl/dist/glsl_version';
import { getKernel } from '@tensorflow/tfjs-core/dist/kernel_registry';

const glsl = getGlslDifferences();
var ks = 7;
let halfk = Math.floor(ks / 2)

let dil = 1;
let hdk = Math.floor((ks / dil) / 2)
let dk = ks / dil

let height = 436 / 4
let width = 1024 / 4

class App extends React.Component {
  constructor(props) {
    super(props);
    this.inited = false;

  }
  getPrepModel = () => {
    let tf = window.tf;


    let consub2 = class consub extends tf.layers.Layer {
      constructor(config) {
        super(config);
        this.axis = config.axis;
      }
      computeOutputShape() {
        return [null, height - ks + 1, width - ks + 1, 3]
      }
      build(inputShape) { this.filter = tf.fill([dk, dk, 3, 1], 1 / (dk * dk)) }
      call(input) {
        return tf.tidy(() => {
          let len = dk * dk
          let strides = [dk, dk]
          let conv = tf.conv2d(tf.abs(tf.concat(input, -4)), this.filter, strides, 'valid',);
          let unst = tf.unstack(conv)
          return tf.concat(unst, -1).reshape([1, 14, 35, dk * dk])
        });
      }
      getConfig() {
        const config = super.getConfig();
        Object.assign(config, { axis: this.axis });
        return config;
      }
      static get className() {
        return 'consub';
      }
    }

    tf.serialization.registerClass(consub2);

    const input1 = tf.input({ name: 'imgcurrent', shape: [height - hdk, width - hdk, 3] });
    const input2 = tf.input({ name: 'imgnext', shape: [height - hdk, width - hdk, 3] });

    let cropInp = tf.layers.cropping2D({ cropping: [[halfk, halfk], [halfk, halfk]] }).apply(input1)
    let layers = []
    let x = 0;

    for (let y = 0; y !== ks; y += dil) {
      for (let x = 0; x !== ks; x += dil) {
        let crop = tf.layers.cropping2D({ cropping: [[y, ks - 1 - y], [x, ks - 1 - x]] }).apply(input2)

        let diff = tf.layers.add().apply([cropInp, crop])
        layers.push(diff)
      }
    }

    let diffs = (new consub2({ axis: -2 })).apply((layers))
    console.log(diffs.shape)

    //let kernels = tf.layers.conv2d({filters:1,kernelSize:[dk,dk],strides:[dk,dk],useBias:false,trainable:false,weights:[tf.fill([dk,dk,3,1],1/(dk*dk))]}).apply(diffs)
    return tf.model({ inputs: [input1, input2], outputs: diffs })
  }
  initClass = async () => {
    let tf = window.tf;
    if (this.inited) {
      return;
    }
    this.inited = true;
    let textElement = document.getElementById('log')
    let log = (txt) => {
      textElement.innerHTML += (txt + '\n');
    }
    log('started')
    try {
      //let modelOpticalFlow = await tf.loadLayersModel('/NormalMapTFJS/ks7_dil1_15_0.009646461345255375/model.json');
      //let modelPrep = this.getPrepModel()
      let video;
      video = await new Promise((res) => { let img = new Image(); img.onload = () => res(img); img.src = '/NormalMapTFJS/frame_0002.png' })
      let video2 = await new Promise((res) => { let img = new Image(); img.onload = () => res(img); img.src = '/NormalMapTFJS/frame_0001.png' })
      //video = document.getElementById('video')
      //let stream = await navigator.mediaDevices.getUserMedia({ video: { aspectRatio: { ideal: 0.5 }, facingMode: "environment" } })
      //video.srcObject = stream;
      //await video.play()
      let canvas = document.getElementById('result')
      let canvas2 = document.getElementById('result2')


      let height = video.videoHeight || video.naturalHeight
      let width = video.videoWidth || video.naturalWidth
      console.log('height: ', height, width);
      let current = tf.zeros([1, height, width, 3]);
      let pred = tf.zeros([1, height, width, 3]);
      let current2 = tf.zeros([1, height / 2, width / 2, 3]);
      let pred2 = tf.zeros([1, height / 2, width / 2, 3]);
      let current4 = tf.zeros([1, height / 4, width / 4, 3]);
      let pred4 = tf.zeros([1, height / 4, width / 4, 3]);

      let backend = tf.backend()

      let uv = tf.tensor([0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0,
        0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1,
        0, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2,
        0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3,
        0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4,
        0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5,
        0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 6, 6, 6], [7, 7, 2]).sub(3)

      const squareAndAddKernel = inputShape => {
        return {
          variableNames: ['PRED', 'NEXT'],
          outputShape: inputShape.slice(),
          userCode: `
          void main() {
            ivec4 coords = getOutputCoords();
            //14, 35, 49
            int dx = coords.z;
            int dy = coords.w;
            float acc = 0.0;
            for(int y=0;y!=7;y++){
              for(int x = 0;x!=7;x++){
                float r = getPRED(coords.x*4+3+x,coords.y*4+3+y,0);
                float g = getPRED(coords.x*4+3+x,coords.y*4+3+y,1);
                float b = getPRED(coords.x*4+3+x,coords.y*4+3+y,2);
                acc+=abs(r-getNEXT(coords.x*4+dx+x,coords.y*4+dy+y,0));
                acc+=abs(g-getNEXT(coords.x*4+dx+x,coords.y*4+dy+y,1));
                acc+=abs(b-getNEXT(coords.x*4+dx+x,coords.y*4+dy+y,2));
              }
            }
            setOutput(acc/(49.0*1.0));
            }
        `
        }
      }
      const findKernel = (inputShape, backSize) => {
        return {
          variableNames: ['BACK', 'PRED', 'NEXT'],
          outputShape: inputShape.slice(),
          userCode: `
          void main() {
            ivec4 coords = getOutputCoords();
            //14, 35, 49
            int dx = coords.z;
            int dy = coords.w;

            int cx = coords.x;
            int addX = int(getBACK(int((float(coords.x)/${inputShape[0]}.0)*${backSize[0]}.0),int((float(coords.y)/${inputShape[1]}.0)*${backSize[1]}.0),0));
            int cy = coords.y;
            int addY = int(getBACK(int((float(coords.x)/${inputShape[0]}.0)*${backSize[0]}.0),int((float(coords.y)/${inputShape[1]}.0)*${backSize[1]}.0),1));
            float acc = 0.0;
            for(int y=0;y!=7;y++){
              for(int x = 0;x!=7;x++){
                float r = getPRED(cx*4+3+x,cy*4+3+y,0);
                float g = getPRED(cx*4+3+x,cy*4+3+y,1);
                float b = getPRED(cx*4+3+x,cy*4+3+y,2);
                acc+=abs(r-getNEXT(cx*4+addY+dx+x,cy*4+addX+dy+y,0));
                acc+=abs(g-getNEXT(cx*4+addY+dx+x,cy*4+addX+dy+y,1));
                acc+=abs(b-getNEXT(cx*4+addY+dx+x,cy*4+addX+dy+y,2));
              }
            }
            setOutput((acc)/(49.0*1.0));
            }
        `
        }
      }
      const blurKernel = inputShape => {
        return {
          variableNames: ['IMG'],
          outputShape: inputShape.slice(),
          userCode: `
          void main() {
            ivec3 coords = getOutputCoords();

            int d = coords.z;
            float acc = 0.0;
            
            acc+=getIMG(coords.x*2-1,coords.y*2-1,d);
            acc+=getIMG(coords.x*2-0,coords.y*2-1,d);
            acc+=getIMG(coords.x*2+1,coords.y*2-1,d);
            acc+=getIMG(coords.x*2-1,coords.y*2-0,d);
            acc+=getIMG(coords.x*2-0,coords.y*2-0,d);
            acc+=getIMG(coords.x*2+1,coords.y*2-0,d);
            acc+=getIMG(coords.x*2-1,coords.y*2+1,d);
            acc+=getIMG(coords.x*2-0,coords.y*2+1,d);
            acc+=getIMG(coords.x*2+1,coords.y*2+1,d);
            
            
            setOutput(acc/(9.0));
            }
        `
        }
      }

      let update = () => {
        let result = null;
        let time = new Date()
        try {

          // if (pred) {
          //   pred.dispose();
          // }
          // pred = current;

          current2 = tf.browser.fromPixels(video).expandDims().div(255)

          let temp = current2;
          current2 = current2.resizeBilinear([height / 2, width / 2])
          temp.dispose();
          temp = current4
          current4 = current2.resizeBilinear([height / 4, width / 4])
          temp.dispose();

          pred2 = tf.browser.fromPixels(video2).expandDims().div(255)
          temp = pred2;
          pred2 = pred2.resizeBilinear([height / 2, width / 2])
          temp.dispose();
          temp = pred4;
          pred4 = pred2.resizeBilinear([height / 4, width / 4])
          temp.dispose();

          result = tf.tidy(() => {

            let x = 0;
            let y = 0;

            let px = Math.floor((current4.shape[2] - 7) / 4)
            let py = Math.floor((current4.shape[1] - 7) / 4)
            let dx = Math.floor(px / 2)
            let dy = Math.floor(py / 2)
            let t;

            let result = backend.compileAndRun(findKernel([py, px, 7, 7], [2, 2]), [tf.tensor([0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2]), pred4, current4]);
            result = result.reshape([result.shape[0], result.shape[1], 7 * 7])

            t = (result.reshape([1, py, px, 49]))

            let shift = t.reshape([py * px, 7, 7, 1]).mean(-4)
            let shiftRes = shift.reshape([7 * 7]).argMin().dataSync()[0]
            x = Math.floor(shiftRes % 7) - 3
            y = Math.floor(shiftRes / 7) - 3

            result = backend.compileAndRun(findKernel([py, px, 7, 7], [2, 2]), [tf.tensor([x, y, x, y, x, y, x, y], [2, 2, 2]), pred4, current4]);
            t = (result.reshape([1, py, px, 49]))

            let program2 = blurKernel([dy, dx, 49]);
            result = backend.compileAndRun(program2, [result.reshape([py, px, 49])]);
            t = (result.reshape([1, dy, dx, 49]))

            let pos = t.reshape([t.shape[1], t.shape[2], 49]).argMin(-1)

            let xp = pos.mod(dk).sub(3).expandDims(-1).add(x);
            let yp = pos.floorDiv(dk).sub(3).expandDims(-1).add(y);

            let out = tf.concat([xp, yp, tf.zeros(xp.shape)], -1).mul(27).add(127).maximum(0).minimum(255).toInt();
            tf.browser.toPixels(out, canvas)

            // 2 step
            let back2 = tf.concat([xp, yp], -1).resizeBilinear([py,px]).mul(2)
            px = Math.floor((current2.shape[2] - 7) / 4)
            py = Math.floor((current2.shape[1] - 7) / 4)
            dx = Math.floor(px / 2)
            dy = Math.floor(py / 2)

            result = backend.compileAndRun(findKernel([py, px, 7, 7], back2.shape.slice(0, 2)), [back2, pred2, current2]);
            t = (result.reshape([1, py, px, 49]))

            program2 = blurKernel([dy, dx, 49]);
            result = backend.compileAndRun(program2, [result.reshape([py, px, 49])]);
            t = (result.reshape([1, dy, dx, 49]))

            pos = t.reshape([t.shape[1], t.shape[2], 49]).argMin(-1)

            xp = pos.mod(dk).sub(3).expandDims(-1);
            yp = pos.floorDiv(dk).sub(3).expandDims(-1);

            out = tf.concat([xp, yp], -1).add(back2.resizeBilinear(xp.shape.slice(0, 2))).concat([tf.zeros(xp.shape)], -1).mul(18).add(127).maximum(0).minimum(255).toInt();

            tf.browser.toPixels(out, canvas2)
            //console.log('out: ', out);
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

      requestAnimationFrame(update)
    } catch (err) {
      console.log('err: ', err);
      log(err.message)
      log(err.stack)
    }
  }
  render() {
    console.log(this);
    return (
      <div className="App">
        <video id='video' style={{ maxHeight: '200px' }}></video>
        <canvas id='result' style={{ width: '30%' }} ></canvas>
        <canvas id='result2' style={{ width: '30%' }} ></canvas>
        <div id="but" onClick={() => {
          this.initClass().catch(err => { console.error(err); document.body.innerText = err.message })
        }}>PlayClass</div>

        <pre id="log">

        </pre>
      </div>
    );
  }
}

export default App;

/*
import { getGlslDifferences } from '../../glsl_version';
export class FromPixelsPackedProgram {
    constructor(outputShape) {
        this.variableNames = ['A'];
        this.packedInputs = false;
        this.packedOutput = true;
        const glsl = getGlslDifferences();
        const [height, width,] = outputShape;
        this.outputShape = outputShape;
        this.userCode = `
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${width}.0, ${height}.0);
            vec4 values = ${glsl.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${glsl.output} = result;
      }
    `;
    }
}
*/