import React from 'react';
import ReactDOM from 'react-dom';
import * as ReactRouter from 'react-router-dom'
import './index.css';
import NormalMap from './AppClass';
import OpticalFlow from './OpticalFlow';
import reportWebVitals from './reportWebVitals';

let { BrowserRouter, Switch,
  Route,
  Link
} = ReactRouter
ReactDOM.render(
  <React.StrictMode>
    <BrowserRouter>
      <div>
        <nav>
          <ul>
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/normalMap">normalMap</Link>
            </li>
            <li>
              <Link to="/opticalFlow">opticalFlow</Link>
            </li>
          </ul>
        </nav>
        <Switch>
          <Route path="/normalMap">
            <NormalMap />
          </Route>
          <Route path="/opticalFlow">
            <OpticalFlow />
          </Route>
          <Route path="/">
          </Route>
        </Switch>
      </div>
    </BrowserRouter>
  </React.StrictMode>,
  document.getElementById('root')
);


// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
