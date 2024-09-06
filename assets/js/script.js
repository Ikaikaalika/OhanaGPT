// Mock user database
const users = [];

// Current logged-in user (for demo purposes)
let currentUser = null;

// Show register form
document.getElementById('showRegister').addEventListener('click', function(event) {
    event.preventDefault();
    document.querySelector('.login-container').style.display = 'none';
    document.querySelector('.register-container').style.display = 'block';
});

// Show login form
document.getElementById('showLogin').addEventListener('click', function(event) {
    event.preventDefault();
    document.querySelector('.register-container').style.display = 'none';
    document.querySelector('.login-container').style.display = 'block';
});

// Handle registration
document.getElementById('registerForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const username = document.getElementById('registerUsername').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const privacyAgreement = document.getElementById('privacyAgreement').checked;
    const usernameMessage = document.getElementById('usernameMessage').textContent;
    const passwordStrength = document.getElementById('passwordStrength').textContent;

    // Prevent registration if username is taken or password is weak
    if (usernameMessage === "Username is already taken") {
        alert('Username is already taken. Please choose another one.');
        return;
    }

    if (passwordStrength === "Weak") {
        alert('Your password is too weak. Please choose a stronger password.');
        return;
    }

    if (password !== confirmPassword) {
        alert('Passwords do not match');
        return;
    }

    if (!privacyAgreement) {
        alert('You must agree to the privacy policy to register');
        return;
    }

    const existingUser = users.find(user => user.username === username);
    if (existingUser) {
        alert('Username already taken');
        return;
    }

    users.push({ username, password });
    alert('Registration successful! Please log in.');

    document.querySelector('.register-container').style.display = 'none';
    document.querySelector('.login-container').style.display = 'block';
});

// Handle username uniqueness check
document.getElementById('registerUsername').addEventListener('input', function() {
    const username = document.getElementById('registerUsername').value;
    const usernameMessage = document.getElementById('usernameMessage');

    if (username.length === 0) {
        usernameMessage.textContent = '';
        return;
    }

    fetch(`/check-username?username=${username}`)
        .then(response => response.json())
        .then(data => {
            if (data.available) {
                usernameMessage.textContent = 'Username is available';
                usernameMessage.style.color = 'green';
            } else {
                usernameMessage.textContent = 'Username is already taken';
                usernameMessage.style.color = 'red';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

/// Handle password strength check
document.getElementById('registerPassword').addEventListener('input', function() {
    const password = document.getElementById('registerPassword').value;
    const passwordStrength = document.getElementById('passwordStrength');

    const strength = getPasswordStrength(password);

    // Update password strength indicator
    if (strength === 'Weak') {
        passwordStrength.textContent = 'Weak';
        passwordStrength.style.color = 'red';
    } else if (strength === 'Moderate') {
        passwordStrength.textContent = 'Moderate';
        passwordStrength.style.color = 'orange';
    } else if (strength === 'Strong') {
        passwordStrength.textContent = 'Strong';
        passwordStrength.style.color = 'green';
    }
});

// Function to determine password strength
function getPasswordStrength(password) {
    if (password.length < 6) {
        return 'Weak';
    }

    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    if (hasUpperCase && hasLowerCase && hasNumbers && hasSpecial && password.length >= 8) {
        return 'Strong';
    } else if (hasUpperCase || hasLowerCase || hasNumbers || hasSpecial) {
        return 'Moderate';
    } else {
        return 'Weak';
    }
}

// Handle login
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;

    const user = users.find(user => user.username === username && user.password === password);
    if (user) {
        currentUser = user;
        document.querySelector('.login-container').style.display = 'none';
        document.querySelector('.upload-container').style.display = 'block';
        document.getElementById('settingsMenu').style.display = 'block';
    } else {
        alert('Invalid login credentials');
    }
});

// Handle file upload
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (file && file.name.endsWith('.ged')) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
          .then(data => {
              if (data.success) {
                  alert('File uploaded and saved to database successfully');
                  // Clear the file input after successful upload
                  fileInput.value = '';
              } else {
                  alert('File upload failed');
              }
          }).catch(error => {
              console.error('Error:', error);
              alert('An error occurred during file upload');
          });
    } else {
        alert('Please select a valid .GED file');
    }
});

// Handle account deletion
document.getElementById('deleteAccount').addEventListener('click', function() {
    if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
        fetch('/delete-account', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username: currentUser.username })
        }).then(response => response.json())
          .then(data => {
              if (data.success) {
                  alert('Your account and data have been deleted.');
                  document.getElementById('settingsMenu').style.display = 'none';
                  document.querySelector('.upload-container').style.display = 'none';
                  document.querySelector('.login-container').style.display = 'block';
                  currentUser = null;
              } else {
                  alert('There was an error deleting your account.');
              }
          }).catch(error => {
              console.error('Error:', error);
              alert('An error occurred while deleting your account.');
          });
    }
});