// Common animations and interactions
document.addEventListener('DOMContentLoaded', function() {
  // Add hover effect to buttons
  const buttons = document.querySelectorAll('.btn');
  buttons.forEach(button => {
      button.addEventListener('mouseenter', function() {
          this.style.transform = 'translateY(-2px)';
      });
      button.addEventListener('mouseleave', function() {
          this.style.transform = 'translateY(0)';
      });
  });
  
  // Connect Google Fit button effect
  const connectButton = document.getElementById('connectButton');
  if (connectButton) {
      connectButton.addEventListener('click', function(e) {
          e.preventDefault();
          
          // Loading animation
          this.innerHTML = 'Connecting...';
          this.classList.add('loading');
          
          // Simulate connection process
          setTimeout(() => {
              this.innerHTML = 'Connected!';
              this.classList.remove('loading');
              this.classList.add('connected');
              
              // Redirect after success
              setTimeout(() => {
                  window.location.href = 'dashboard.html';
              }, 1000);
          }, 2000);
      });
  }
  
  // Animate elements when they come into view
  const animateOnScroll = function() {
      const elements = document.querySelectorAll('.feature-card, .benefit-item');
      
      elements.forEach(element => {
          const elementPosition = element.getBoundingClientRect().top;
          const windowHeight = window.innerHeight;
          
          if (elementPosition < windowHeight - 100) {
              element.style.animationPlayState = 'running';
          }
      });
  };
  
  window.addEventListener('scroll', animateOnScroll);
  animateOnScroll(); // Run once on load
});