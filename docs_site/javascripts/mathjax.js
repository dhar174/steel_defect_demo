// MathJax configuration for Steel Defect Prediction System documentation
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Additional JavaScript for enhanced functionality
document.addEventListener('DOMContentLoaded', function() {
  // Add copy button to code blocks
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach(function(block) {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.innerHTML = '📋 Copy';
    button.onclick = function() {
      navigator.clipboard.writeText(block.textContent);
      button.innerHTML = '✅ Copied!';
      setTimeout(() => {
        button.innerHTML = '📋 Copy';
      }, 2000);
    };
    block.parentNode.appendChild(button);
  });

  // Enhanced table of contents
  const tocLinks = document.querySelectorAll('.md-nav__link');
  tocLinks.forEach(function(link) {
    link.addEventListener('click', function(e) {
      // Smooth scroll to section
      const href = this.getAttribute('href');
      if (href.startsWith('#')) {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });
  });
});