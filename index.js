
const express = require('express')
const app = express()
let https = require('https')
let path = require('path')
const port = 3000
let fs = require('fs')
var key = fs.readFileSync('./httpsKeys/selfsigned.key');
var cert = fs.readFileSync('./httpsKeys/selfsigned.crt');
var options = {
  key: key,
  cert: cert
};
app.get('/', (req, res) => {
  res.sendFile(path.resolve('./demo.html'))
})
app.use('/model',express.static('model'));

var server = https.createServer(options, app);
server.listen(port, () => {
  console.log(`App listening at http://localhost:${port}`)
})