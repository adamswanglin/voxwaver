import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/app.css'

// macOS traffic lights overlay the titlebar: reserve space for them
if (navigator.userAgent.includes('Mac')) document.documentElement.classList.add('mac')

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
