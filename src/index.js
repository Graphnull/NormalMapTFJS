import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// ReactDOM.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>,
//   document.getElementById('root')
// );

document.getElementById('root').innerHTML=`<video id='video'></video>
<canvas id='result'></canvas>
<script>
    let inited = false
    let init = (async () => {

        if (inited) {
            return;
        }
        inited = true;
        let textElement = document.getElementById('but')
        textElement.innerHTML = 'started'
        console.log(tf)
        //await tf.setBackend('cpu')
        let model = await tf.loadLayersModel('/model/model.json');
        model.inputs[0].shape = [null,null,null,3]
        model.inputLayers[0].batchInputShape = [null,null,null,3]
        model.inputLayers[0].inputSpec[0].shape = [null,null,null,3]
        model.feedInputShapes[0] = [null,null,null,3]
        model.internalInputShapes[0] = [null,null,null,3]
        console.log('model: ', model);
        textElement.innerHTML = 'loaded'
        console.log(model);
        let stream = await navigator.mediaDevices.getUserMedia({ video: {width:512, height:512} })

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
                        image = tf.image.resizeBilinear(image, [256, 256])
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

    })

</script>
<div id="but" onclick="init().catch(err=>{document.body.innerText= err.message})">play</div>`

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
