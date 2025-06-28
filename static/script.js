 document.addEventListener('DOMContentLoaded', () => {
      const sendBtn = document.getElementById('send-btn');
      const userInput = document.getElementById('user-input');

      function addMessage(sender, message) {
        const chatBox = document.getElementById('chat-box');
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        msgDiv.innerHTML = `
          <div class="avatar">${sender === 'user' ? '👤' : '🤖'}</div>
          <div class="bubble">${message}</div>
        `;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
      }

      sendBtn.addEventListener('click', async () => {
        const message = userInput.value.trim();
        if (!message) return;

        addMessage('user', message);
        userInput.value = '';

        try {
          const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
          });

          const data = await response.json();

          if (response.ok) {
            addMessage('bot', data.response);
          } else {
            addMessage('bot', data.error || 'Something went wrong.');
          }
        } catch (err) {
          console.error("Chat failed:", err);
          addMessage('bot', 'Connection failed. Try again later.');
        }
      });

      window.location.href = "/chat";

      // Handle file selection
document.getElementById('file-upload').addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (file) {
    console.log("File selected:", file.name);
    alert("Uploading: " + file.name); // Replace with real upload logic later
  }
});

      // Optional: Press Enter to Send Message
      userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          sendBtn.click();
        }
      });
    });

// document.addEventListener('DOMContentLoaded', () => {
//   const sendBtn = document.getElementById('send-btn');
//   const userInput = document.getElementById('user-input');

//   function addMessage(sender, message) {
//     const chatBox = document.getElementById('chat-box');
//     const msgDiv = document.createElement('div');
//     msgDiv.className = `message ${sender}`;
//     msgDiv.innerHTML = `
//       <div class="avatar">${sender === 'user' ? '👤' : '🤖'}</div>
//       <div class="bubble">${message}</div>
//     `;
//     chatBox.appendChild(msgDiv);
//     chatBox.scrollTop = chatBox.scrollHeight;
//   }

//   sendBtn.addEventListener('click', async () => {
//     const message = userInput.value.trim();
//     if (!message) return;

//     addMessage('user', message);
//     userInput.value = '';

//     try {
//       const response = await fetch('/chat', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ message })
//       });

//       const data = await response.json();

//       if (response.ok) {
//         addMessage('bot', data.response);
//       } else {
//         addMessage('bot', data.error || 'Something went wrong.');
//       }
//     } catch (err) {
//       console.error("Chat failed:", err);
//       addMessage('bot', 'Connection failed. Try again later.');
//     }
//   });

//   // Handle file upload
//   document.getElementById('file-upload').addEventListener('change', function (e) {
//     const file = e.target.files[0];
//     if (file) {
//       console.log("File selected:", file.name);
//       alert("Uploading: " + file.name); // Replace with real upload logic
//     }
//   });

//   // Press Enter to Send Message
//   userInput.addEventListener('keypress', (e) => {
//     if (e.key === 'Enter') {
//       sendBtn.click();
//     }
//   });
// });