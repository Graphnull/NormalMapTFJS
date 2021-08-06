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
      let modelOpticalFlow = await tf.loadLayersModel('/NormalMapTFJS/ks7_dil1_25_0.007120296359062195/model.json');
      let modelPrep = this.getPrepModel()
      let video;
      // video = document.getElementById('video')
      //video.srcObject = stream;
      //await video.play()
      let canvas = document.getElementById('result')
      let canvas2 = document.getElementById('result2')
      let current = tf.zeros([1, height, width, 3]);
      let pred = tf.zeros([1, height, width, 3]);

      let backend = tf.backend()
      let ENGINE = tf.engine();
      let gl = backend.gpgpu.gl;

      video = await new Promise((res) => { let img = new Image(); img.onload = () => res(img); img.src = '/NormalMapTFJS/test.png' })
      let uv = tf.tensor([0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0,
        0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1,
        0, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2,
        0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3,
        0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4,
        0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5,
        0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 6, 6, 6], [7, 7, 2]).sub(3)
      let zeros = tf.zeros([14, 35, 1]);
      let ctime = 0;


      let nv = false;
      const squareAndAddKernel = inputShape => {
        let [count, height, width, channels] = inputShape
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
                float r = getPRED(coords.x*7+3+x,coords.y*7+3+y,0);
                float g = getPRED(coords.x*7+3+x,coords.y*7+3+y,1);
                float b = getPRED(coords.x*7+3+x,coords.y*7+3+y,2);
                acc+=(r+getNEXT(coords.x*7+dx+x,coords.y*7+dy+y,0));
                acc+=(g+getNEXT(coords.x*7+dx+x,coords.y*7+dy+y,1));
                acc+=(b+getNEXT(coords.x*7+dx+x,coords.y*7+dy+y,2));
              }
            }
            setOutput(abs(acc)/(49.0*1.0));

            }
        `
        }
      }


      let update = () => {
        let result = null;
        let time = new Date()
        try {

          if (pred) {
            pred.dispose();
          }
          pred = current;

          current = tf.browser.fromPixels(video).expandDims()
          //log([current.shape[2], current.shape[1], width, height])
          //480 313 256 109
          if (current.shape[2] !== width || current.shape[1] !== height) {
            let temp = current;
            current = current.resizeBilinear([height, width])
            temp.dispose();
          }


          result = tf.tidy(() => {
            //log('req1')
            let nw = Math.floor((width - ks - 1) / dk)
            let nh = Math.floor((height - ks - 1) / dk)
            let tensorc = pred.slice([0, 0, 0, 0], [1, height - hdk, width - hdk, 3])
            let tensorn = current.slice([0, 0, 0, 0], [1, height - hdk, width - hdk, 3]).div(-1)

            //let time = new Date()
            //console.log(tensorc, tensorn, modelPrep.predict([tensorc, tensorn]).dataSync());
            //console.log(new Date() - time)


            let t;
            if (nv) {
              t = tf.zeros([1, 14, 35, 3], 'float32')

              let program1 = squareAndAddKernel([14, 35, 7, 7]);
              let result = backend.compileAndRun(program1, [tensorc, tensorn]);
              //console.log('result: ', result);
              t = result.reshape([1, 14, 35, 49])
              //result = result.reshape([14, 35, 49]).slice([0, 0, 0], [14, 35, 3]).maximum(0).minimum(255).toInt()//.slice([0, 0, 3], [14, 35, 3]);
              //tf.browser.toPixels(result, canvas)
            } else {
              t = modelPrep.predict([tensorc, tensorn]).reshape([14, 35, 49])//.slice([0, 0, 0], [14, 35, 3]).maximum(0).minimum(255).toInt()

              //tf.browser.toPixels(result, canvas2)
            }

            // 1, 14,35,49
            //console.log(t)
            let res = modelOpticalFlow.predict(t)
            let sum = res.reshape([nh, nw, dk * dk, 1]).relu().pow(2).sum(-2)

            let p = res.reshape([nh, nw, dk, dk, 1])

            p = p.concat([p], -1).pow(2).mul(uv).sum(-2).sum(-2)

            sum = sum.concat([sum], -1)
            let result2 = tf.concat([p.div(sum), zeros], -1).mul(3)
            return result2.mul(9).add(127).maximum(0).minimum(255).toInt();

          })
        } catch (err) {
          console.log('err: ', err);

          log(err.message)
        }
        //console.log(result, canvas);
        tf.browser.toPixels(result, canvas).then(() => {
          ctime = ctime * 0.9 + (new Date() - time) * 0.1
          console.log('time', ctime)
          //log('req3')
          result.dispose();
          requestAnimationFrame(update)
        }).catch(err => {
          console.log('err: ', err);
          log(err.message)
        })
      }

      requestAnimationFrame(update)
    } catch (err) {
      console.log('err: ', err);
      log(err.message)
    }
  }
  render() {
    console.log(this);
    return (
      <div className="App">
        <video id='video' style={{ maxHeight: '200px' }}></video>
        <canvas id='result' style={{ width: '20%' }} ></canvas>
        <canvas id='result2' style={{ width: '20%' }} ></canvas>
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