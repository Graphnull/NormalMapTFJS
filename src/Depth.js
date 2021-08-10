import React from 'react'
import logo from './logo.svg';
import './App.css';

class App extends React.Component {
  constructor(props){
    super(props);
    this.inited = false;

  }
  init = async () => {
        let tf = window.tf;
        if (this.inited) {
            return;
        }
        this.inited = true;
        let textElement = document.getElementById('but')
        textElement.innerHTML = 'started'
        let model = await tf.loadLayersModel('/NormalMapTFJS/model/model.json');
        model.inputs[0].shape = [null,null,null,3]
        model.inputLayers[0].batchInputShape = [null,null,null,3]
        model.inputLayers[0].inputSpec[0].shape = [null,null,null,3]
        model.feedInputShapes[0] = [null,null,null,3]
        model.internalInputShapes[0] = [null,null,null,3]

        textElement.innerHTML = 'loaded'
        let stream = await navigator.mediaDevices.getUserMedia({ video: {width:512, height:512},facingMode: "environment" })

        textElement.innerHTML = 'capture'
        let video = document.getElementById('video')
        video.srcObject = stream;
        await video.play()
        let canvas = document.getElementById('result')
        let ctx = canvas.getContext('2d');

        let update = () => {
            let result = tf.tidy(() => {
                    textElement.innerHTML = 'req'
                try {
                    let image = tf.browser.fromPixels(video).expandDims().div(255)
                    //if(image.shape[2]!==1024|| image.shape[1]!==768){
                        image = tf.image.resizeBilinear(image, [512, 512])
                    //}
                    textElement.innerHTML = 'req1'
                    let result = model.predict(image);
                    textElement.innerHTML = 'req2'
                    result =  result.reshape(result.shape.slice(1)).mul(0.5).add(0.5).maximum(0).minimum(1)
                    return result;
                } catch (err) {

                    textElement.innerHTML = err.message
                }
            })
            tf.browser.toPixels(result, canvas).then(()=>{
                        textElement.innerHTML = 'req3'
                        result.dispose();
                        requestAnimationFrame(update)
                    }).catch(err=>{
                    textElement.innerHTML = err.message
                    })
        }

        requestAnimationFrame(update)

    }
    initDepth = async () => {
      let tf = window.tf;
      if (this.inited) {
          return;
      }
      this.inited = true;
      let textElement = document.getElementById('but')
      textElement.innerHTML = 'started'
      let model = await tf.loadLayersModel('/NormalMapTFJS/depth/model.json');
      // model.inputs[0].shape = [null,null,null,3]
      // model.inputLayers[0].batchInputShape = [null,null,null,3]
      // model.inputLayers[0].inputSpec[0].shape = [null,null,null,3]
      // model.feedInputShapes[0] = [null,null,null,3]
      // model.internalInputShapes[0] = [null,null,null,3]

      textElement.innerHTML = 'loaded'
      let stream = await navigator.mediaDevices.getUserMedia({ video: {width:256, height:192},facingMode: "environment" })

      textElement.innerHTML = 'capture'
      let video = document.getElementById('video')
      video.srcObject = stream;
      await video.play()
      let canvas = document.getElementById('result')
      let ctx = canvas.getContext('2d');
      tf.enableProdMode()
      let update = () => {
          let result = tf.tidy(() => {
                  textElement.innerHTML = 'req'
              try {
                let time = new Date()
                  let image = tf.browser.fromPixels(video).expandDims().div(255)
                  //if(image.shape[2]!==1024|| image.shape[1]!==768){192,256
                      image = tf.image.resizeBilinear(image, [192, 256])
                  //}
                  textElement.innerHTML = 'req1'
                  let result = model.predict(image);
                  textElement.innerHTML = 'req2'
                  let start = [0,8,8,0]
                  let end = [1,result.shape[1]-16,result.shape[2]-16,1]
                  let minV = result.slice(start,end).min().dataSync()[0]
                  let maxV = result.slice(start,end).max().dataSync()[0]-minV
                  result =  result.reshape(result.shape.slice(1)).sub(minV).div(maxV).maximum(0).minimum(1)
                  
                  console.log('minV: ', minV, maxV,  new Date()-time);
                  return result;
              } catch (err) {
                console.log('err: ', err);

                  textElement.innerHTML = err.message
              }
          })
          tf.browser.toPixels(result, canvas).then(()=>{
                      textElement.innerHTML = 'req3'
                      result.dispose();
                      requestAnimationFrame(update)
                  }).catch(err=>{
                  textElement.innerHTML = err.message
                  })
      }

      requestAnimationFrame(update)

  }
  render(){
    console.log(this);
    return (
      <div className="App">
        <video id='video'></video>
        <canvas id='result'></canvas>
        <div id="but" onClick={()=>{
          this.initDepth().catch(err=>{console.error(err);document.body.innerText= err.message})
        }}>Play</div>
      </div>
    );
  }
}

export default App;
