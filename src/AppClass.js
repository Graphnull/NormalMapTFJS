import React from 'react'
import logo from './logo.svg';
import './App.css';



class App extends React.Component {
  constructor(props) {
    super(props);
    this.inited = false;

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
      let model = await tf.loadLayersModel('/NormalMapTFJS/modelClassOrient/model.json');
      console.log(model);
      let w = 64
      let h = 64
      model.inputs[0].shape = [null, h * 8, w * 8, 3]
      model.inputLayers[0].batchInputShape = [null, h * 8, w * 8, 3]
      model.inputLayers[0].inputSpec[0].shape = [null, h * 8, w * 8, 3]
      model.feedInputShapes[0] = [null, h * 8, w * 8, 3]
      model.internalInputShapes[0] = [null, h * 8, w * 8, 3]
      model.inputs[1].shape = [null, h / 16, w / 16, 1]
      model.inputLayers[1].batchInputShape = [null, h / 16, w / 16, 1]
      model.inputLayers[1].inputSpec[0].shape = [null, h / 16, w / 16, 1]
      model.feedInputShapes[1] = [null, h / 16, w / 16, 1]
      model.internalInputShapes[1] = [null, h / 16, w / 16, 1]
      if (
        DeviceMotionEvent &&
        typeof DeviceMotionEvent.requestPermission === "function"
      ) {
        DeviceMotionEvent.requestPermission();
      }

      let orient = 0
      window.addEventListener("deviceorientation", (event) => {
        orient = (event.beta - 90) / 180
      });

      log('loaded')
      let stream = await navigator.mediaDevices.getUserMedia({ video: { aspectRatio: { ideal: 1 }, facingMode: "environment" } })

      log('capture')
      let video = document.getElementById('video')
      video.srcObject = stream;
      await video.play()
      let canvas = document.getElementById('result')
      let ctx = canvas.getContext('2d');

      let update = () => {
        let result = tf.tidy(() => {
          log('req')
          try {
            let image = tf.browser.fromPixels(video).expandDims().div(255)

            //if(image.shape[2]!==1024|| image.shape[1]!==768){
            image = tf.image.resizeBilinear(image, [512, 512])
            //}
            log('req1')
            let time = new Date()
            let result = model.predict([image, tf.tensor(new Float32Array((h / 16) * (w / 16)).map(v => orient), [1, h / 16, w / 16, 1])]);
            log('req2')
            result = result.reshape(result.shape.slice(1)).slice([0, 0, 0], [h, w, 36])
            let out1 = result.dataSync()
            console.log('t', new Date() - time);
            let viz = new Float32Array(h * w * 3)

            for (let y = 0; y !== h; y++) {
              for (let x = 0; x !== w; x++) {
                let maxv = 0
                let maxs = 0
                for (let i = 0; i !== 18; i++) {
                  if (maxv < out1[y * w * 36 + x * 36 + i]) {
                    maxv += i * Math.pow(Math.max(out1[y * w * 36 + x * 36 + i], 0), 2)
                    maxs += Math.pow(Math.max(out1[y * w * 36 + x * 36 + i], 0), 2)
                  }
                }
                let dx = Math.min(Math.max((maxv / maxs) * 15, 0), 255)
                maxv = 0
                maxs = 0
                for (let i = 0; i !== 18; i++) {
                  if (maxv < out1[y * w * 36 + x * 36 + i + 18]) {
                    maxv += i * Math.pow(Math.max(out1[y * w * 36 + x * 36 + i + 18], 0), 2)
                    maxs += Math.pow(Math.max(out1[y * w * 36 + x * 36 + i + 18], 0), 2)
                  }
                }
                let dy = Math.min(Math.max((maxv / maxs) * 15, 0), 255)

                viz[y * w * 3 + x * 3 + 0] = dx
                viz[y * w * 3 + x * 3 + 1] = dy
                viz[y * w * 3 + x * 3 + 2] = 0
              }
            }
            let rest = tf.tensor(viz, [h, w, 3], 'int32').maximum(0).minimum(255).toInt()

            return rest;
          } catch (err) {
            console.log('err: ', err);

            log(err.message)
          }
        })
        console.log(result, canvas);
        tf.browser.toPixels(result, canvas).then(() => {
          log('req3')
          result.dispose();
          requestAnimationFrame(update)
        }).catch(err => {
          log(err.message)
        })
      }

      requestAnimationFrame(update)
    } catch (err) {
      log(err.message)
    }
  }
  initImg = async () => {
    let tf = window.tf;
    if (this.inited) {
      return;
    }
    this.inited = true;
    let textElement = document.getElementById('but')
    textElement.innerHTML = 'started'
    let model = await tf.loadLayersModel('/NormalMapTFJS/modelClassOrient/model.json');
    console.log(model);
    let w = 64
    let h = 64
    model.inputs[0].shape = [null, h * 8, w * 8, 3]
    model.inputLayers[0].batchInputShape = [null, h * 8, w * 8, 3]
    model.inputLayers[0].inputSpec[0].shape = [null, h * 8, w * 8, 3]
    model.feedInputShapes[0] = [null, h * 8, w * 8, 3]
    model.internalInputShapes[0] = [null, h * 8, w * 8, 3]
    model.inputs[1].shape = [null, h / 16, w / 16, 1]
    model.inputLayers[1].batchInputShape = [null, h / 16, w / 16, 1]
    model.inputLayers[1].inputSpec[0].shape = [null, h / 16, w / 16, 1]
    model.feedInputShapes[1] = [null, h / 16, w / 16, 1]
    model.internalInputShapes[1] = [null, h / 16, w / 16, 1]

    textElement.innerHTML = 'loaded'
    let img = await new Promise((res) => { let img = new Image(); img.onload = () => res(img); img.src = '/NormalMapTFJS/test.png' })


    textElement.innerHTML = 'capture'

    let canvas = document.getElementById('result')
    let ctx = canvas.getContext('2d');


    let update = () => {
      let result = tf.tidy(() => {
        textElement.innerHTML = 'req'
        try {
          let image = tf.browser.fromPixels(img).expandDims().div(255)

          //if(image.shape[2]!==1024|| image.shape[1]!==768){
          image = tf.image.resizeBilinear(image, [h * 8, w * 8])
          //}
          let time = new Date()
          let result = model.predict([image, tf.tensor(new Float32Array((h / 16) * (w / 16)).map(v => 0.1), [1, h / 16, w / 16, 1])]);

          result = result.reshape(result.shape.slice(1)).slice([0, 0, 0], [h, w, 36])
          let out1 = result.dataSync()
          console.log('t', new Date() - time);
          let viz = new Float32Array(h * w * 3)

          for (let y = 0; y !== h; y++) {
            for (let x = 0; x !== w; x++) {
              let maxv = 0;
              let maxs = 0;
              for (let i = 0; i !== 18; i++) {
                if (maxv < out1[y * w * 36 + x * 36 + i]) {
                  maxv += i * Math.pow(Math.max(out1[y * w * 36 + x * 36 + i], 0), 2)
                  maxs += Math.pow(Math.max(out1[y * w * 36 + x * 36 + i], 0), 2)
                }
              }
              let dx = Math.min(Math.max((maxv / maxs) * 15, 0), 255)
              maxv = 0
              maxs = 0
              for (let i = 0; i !== 18; i++) {
                if (maxv < out1[y * w * 36 + x * 36 + i + 18]) {
                  maxv += i * Math.pow(Math.max(out1[y * w * 36 + x * 36 + i + 18], 0), 2)
                  maxs += Math.pow(Math.max(out1[y * w * 36 + x * 36 + i + 18], 0), 2)
                }
              }
              let dy = Math.min(Math.max((maxv / maxs) * 15, 0), 255)

              viz[y * w * 3 + x * 3 + 0] = dx
              viz[y * w * 3 + x * 3 + 1] = dy
              viz[y * w * 3 + x * 3 + 2] = 0
            }
          }
          let rest = tf.tensor(viz, [h, w, 3], 'int32').maximum(0).minimum(255).toInt()

          return rest;
        } catch (err) {
          console.log('err: ', err);

          textElement.innerHTML = err.message
        }
      })
      tf.browser.toPixels(result, canvas).then(() => {
        textElement.innerHTML = 'req3'
        result.dispose();
        requestAnimationFrame(update)
      }).catch(err => {
        textElement.innerHTML = err.message
      })
    }

    requestAnimationFrame(update)

  }
  render() {
    console.log(this);
    return (
      <div className="App">
        <video id='video' style={{ maxHeight: '200px' }}></video>
        <canvas id='result'></canvas>
        <div id="but" onClick={() => {
          this.init().catch(err => { console.error(err); document.body.innerText = err.message })
        }}>Play(v4)</div>

        <div id="but2" onClick={() => {
          this.initImg().catch(err => { console.error(err); document.body.innerText = err.message })
        }}>imgClass</div>

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
