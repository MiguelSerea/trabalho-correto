document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewImage = document.getElementById('previewImage');
    const buttons = document.querySelectorAll('.processing-btn');
    
    let currentFile = null;
  
    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  
    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });
  
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      const files = e.dataTransfer.files;
      handleFile(files[0]);
    });
  
    // Click to upload
    dropZone.addEventListener('click', () => {
      fileInput.click();
    });
  
    fileInput.addEventListener('change', (e) => {
      handleFile(e.target.files[0]);
    });
  
    function handleFile(file) {
      if (file && file.type.startsWith('image/')) {
        currentFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
          previewImage.src = e.target.result;
          previewImage.style.display = 'block';
          enableButtons();
        };
        
        reader.readAsDataURL(file);
      } else {
        alert('Por favor, selecione uma imagem válida.');
      }
    }
  
    function enableButtons() {
      buttons.forEach(button => {
        button.disabled = false;
      });
    }
  
    // Button click handlers
    document.getElementById('centralizedUni').addEventListener('click', () => {
      if (currentFile) {
        console.log('Processando em modo centralizado uniprocesso');
        // Implementar lógica de processamento
      }
    });
  
    document.getElementById('centralizedMulti').addEventListener('click', () => {
      if (currentFile) {
        console.log('Processando em modo centralizado multiprocesso');
        // Implementar lógica de processamento
      }
    });
  
    document.getElementById('decentralizedMulti').addEventListener('click', () => {
      if (currentFile) {
        console.log('Processando em modo descentralizado multiprocesso');
        // Implementar lógica de processamento
      }
    });
  
    document.getElementById('verify').addEventListener('click', () => {
      if (currentFile) {
        console.log('Verificando processamento');
        // Implementar lógica de verificação
      }
    });
  
    // Inicialmente desabilitar os botões
    buttons.forEach(button => {
      button.disabled = true;
    });
  });