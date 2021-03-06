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
        let stream = await navigator.mediaDevices.getUserMedia({ video: {aspectRatio: {ideal: 1},facingMode: "environment" }})

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
  render(){
    console.log(this);
    return (
      <div className="App">
        <video id='video'></video>
        <canvas id='result'></canvas>
        <div id="but" onClick={()=>{
          this.init().catch(err=>{console.error(err);document.body.innerText= err.message})
        }}>Play(v3)</div>
      </div>
    );
  }
}

export default App;
