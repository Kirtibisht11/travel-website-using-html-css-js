
function validateForm() {
  const nameError = "Name can't be blank.";
  const emailError = 'Invalid email format';
  const passwordError = 'Password must be at least 6 characters';
  const phoneError = 'Phone number must be numeric';

  const name = document.myform.name.value.trim();
  const email = document.myform.email.value.trim();
  const password = document.myform.password.value.trim();
  const num = document.myform.num.value.trim();

  let isValid = true;

  // Reset error messages and styles
  document.getElementById('usernameError').textContent = '';
  document.getElementById('emailError').textContent = '';
  document.getElementById('passwordError').textContent = '';
  document.getElementById('phoneError').textContent = '';
  document.myform.name.classList.remove('error');
  document.myform.email.classList.remove('error');
  document.myform.password.classList.remove('error');
  document.myform.num.classList.remove('error');

  // Name validation
  if (name === "") {
    document.getElementById('usernameError').textContent = nameError;
    document.myform.name.classList.add('error');
    isValid = false;
  }

  // Email validation
  const atposition = email.indexOf('@');
  const dotposition = email.lastIndexOf('.');
  if (atposition < 1 || dotposition < atposition + 2 || dotposition + 2 >= email.length) {
    document.getElementById('emailError').textContent = emailError;
    document.myform.email.classList.add('error');
    isValid = false;
  }

  // Password validation
  if (password.length < 6) {
    document.getElementById('passwordError').textContent = passwordError;
    document.myform.password.classList.add('error');
    isValid = false;
  }

  // Phone number validation
  if (isNaN(num)) {
    document.getElementById('phoneError').textContent = phoneError;
    document.myform.num.classList.add('error');
    isValid = false;
  }

  return isValid;
}

