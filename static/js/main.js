// Alternar Visibilidad de Contraseña
function togglePassword(inputId, iconId) {
    const input = document.getElementById(inputId);
    const icon = document.getElementById(iconId);
    if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'bi bi-eye-slash-fill';
    } else {
        input.type = 'password';
        icon.className = 'bi bi-eye-fill';
    }
}

// Cargar Historial desde SQLite
async function loadPersistentHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();
        if (data.records && data.records.length > 0) {
            const tableBody = document.getElementById('tableHistorial');
            if (tableBody) {
                tableBody.innerHTML = '';
                data.records.forEach((rec, idx) => {
                    const tr = document.createElement('tr');
                    tr.className = 'history-row';
                    tr.innerHTML = `
                        <td class="fw-bold">${idx + 1}</td>
                        <td><small>${rec.timestamp}</small></td>
                        <td><small class="text-truncate d-inline-block" style="max-width: 140px;">${rec.filename}</small></td>
                        <td><span class="badge bg-success bg-opacity-10 text-success border border-success border-opacity-20">${rec.diagnosis}</span></td>
                        <td class="fw-semibold">${rec.confidence}</td>
                    `;
                    tableBody.appendChild(tr);
                });
            }
        }
    } catch (err) {
        console.error("Error al cargar historial:", err);
    }
}

// Filtro de Búsqueda en Historial
function filterHistory() {
    const input = document.getElementById('searchHistory');
    if (!input) return;
    const filter = input.value.toLowerCase();
    const rows = document.querySelectorAll('#tableHistorial tr');
    
    rows.forEach(row => {
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(filter) ? '' : 'none';
    });
}

// MANEJO DE SELECCIÓN Y PREVISUALIZACIÓN DE IMAGEN (CLIC Y DRAG & DROP)
function handleFileSelect(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        alert('Por favor selecciona un archivo de imagen válido (JPG, PNG, WEBP).');
        return;
    }

    const reader = new FileReader();
    reader.onload = function (evt) {
        const imagePreview = document.getElementById('imagePreview');
        const previewContainer = document.getElementById('previewContainer');
        const dropZoneDefault = document.getElementById('dropZoneDefault');
        const fileName = document.getElementById('fileName');

        if (imagePreview) imagePreview.src = evt.target.result;
        if (previewContainer) previewContainer.classList.remove('d-none');
        if (dropZoneDefault) dropZoneDefault.classList.add('d-none');
        if (fileName) fileName.innerText = file.name;
    };
    reader.readAsDataURL(file);
}

document.addEventListener('DOMContentLoaded', () => {
    loadPersistentHistory();

    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');

    // Evento de cambio en Input File
    if (fileInput) {
        fileInput.addEventListener('change', function (e) {
            if (e.target.files && e.target.files[0]) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    // Eventos Drag and Drop en DropZone
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('border-success'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('border-success'), false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files[0]) {
                fileInput.files = files; // Asignar al input file
                handleFileSelect(files[0]);
            }
        });
    }
});

// LOGIN AJAX
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', async function (e) {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value.trim();
        const password = document.getElementById('loginPassword').value;
        const alertBox = document.getElementById('loginAlert');
        const btnSubmit = this.querySelector('button[type="submit"]');

        alertBox.className = 'alert alert-info small';
        alertBox.innerHTML = '<div class="spinner-border spinner-border-sm me-2"></div> Validando...';
        alertBox.classList.remove('d-none');
        btnSubmit.disabled = true;

        try {
            const res = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (data.success) {
                alertBox.className = 'alert alert-success small';
                alertBox.innerText = `¡Bienvenido, ${data.user_name}!`;
                setTimeout(() => window.location.reload(), 800);
            } else {
                alertBox.className = 'alert alert-danger small';
                alertBox.innerText = data.error || 'Credenciales incorrectas';
                btnSubmit.disabled = false;
            }
        } catch (err) {
            alertBox.className = 'alert alert-danger small';
            alertBox.innerText = 'Error de conexión';
            btnSubmit.disabled = false;
        }
    });
}

// REGISTER AJAX
const registerForm = document.getElementById('registerForm');
if (registerForm) {
    registerForm.addEventListener('submit', async function (e) {
        e.preventDefault();
        const name = document.getElementById('regName').value.trim();
        const email = document.getElementById('regEmail').value.trim();
        const password = document.getElementById('regPassword').value;
        const alertBox = document.getElementById('registerAlert');
        const btnSubmit = this.querySelector('button[type="submit"]');

        alertBox.className = 'alert alert-info small';
        alertBox.innerHTML = '<div class="spinner-border spinner-border-sm me-2"></div> Registrando...';
        alertBox.classList.remove('d-none');
        btnSubmit.disabled = true;

        try {
            const res = await fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password })
            });
            const data = await res.json();
            if (data.success) {
                alertBox.className = 'alert alert-success small';
                alertBox.innerText = `¡Registro exitoso! Iniciando...`;
                setTimeout(() => window.location.reload(), 1000);
            } else {
                alertBox.className = 'alert alert-danger small';
                alertBox.innerText = data.error || 'Error al registrar';
                btnSubmit.disabled = false;
            }
        } catch (err) {
            alertBox.className = 'alert alert-danger small';
            alertBox.innerText = 'Error de conexión';
            btnSubmit.disabled = false;
        }
    });
}

// LOGOUT
const btnLogout = document.getElementById('btnLogout');
if (btnLogout) {
    btnLogout.addEventListener('click', async function () {
        await fetch('/logout', { method: 'POST' });
        window.location.reload();
    });
}

// INFERENCIA Y RESULTADOS
const uploadForm = document.getElementById('uploadForm');
if (uploadForm) {
    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();
        const fileInput = document.getElementById('fileInput');
        if (!fileInput || !fileInput.files.length) return alert('Selecciona una imagen primero.');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        document.getElementById('loadingSpinner').classList.remove('d-none');
        document.getElementById('resultContainer').classList.add('d-none');
        document.getElementById('btnAnalizar').disabled = true;

        try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            if (res.status === 401) {
                alert('Debes iniciar sesión para ejecutar el análisis.');
                const modal = new bootstrap.Modal(document.getElementById('loginModal'));
                modal.show();
                return;
            }
            const data = await res.json();
            if (data.success) {
                document.getElementById('txtDiagnostico').innerText = data.predicted_class;
                document.getElementById('txtConfianza').innerText = `Confianza: ${data.confidence}`;
                
                const badgeSev = document.getElementById('badgeSeveridad');
                badgeSev.innerText = data.severidad;
                badgeSev.className = `badge fs-6 px-3 py-1.5 rounded-pill font-semibold ms-1 ${data.color_badge}`;

                document.getElementById('imgEntrante').src = data.img_entrada;
                document.getElementById('imgGradCAM').src = data.gradcam;
                document.getElementById('imgChart').src = data.chart;

                const listAcciones = document.getElementById('listAcciones');
                listAcciones.innerHTML = '';
                data.acciones.forEach(acc => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item px-0 border-0 bg-transparent text-slate-700 py-1.5';
                    li.innerHTML = `<i class="bi bi-check-circle-fill text-success me-2"></i>${acc}`;
                    listAcciones.appendChild(li);
                });

                document.getElementById('resultContainer').classList.remove('d-none');
                document.getElementById('btnReset').classList.remove('d-none');

                loadPersistentHistory();
            } else {
                alert('Error: ' + (data.error || 'No se pudo procesar'));
            }
        } catch (err) {
            alert('Error de conexión con el backend.');
        } finally {
            document.getElementById('loadingSpinner').classList.add('d-none');
            document.getElementById('btnAnalizar').disabled = false;
        }
    });
}

// Reset Formulario
const btnReset = document.getElementById('btnReset');
if (btnReset) {
    btnReset.addEventListener('click', function () {
        document.getElementById('uploadForm').reset();
        document.getElementById('previewContainer').classList.add('d-none');
        document.getElementById('dropZoneDefault').classList.remove('d-none');
        document.getElementById('resultContainer').classList.add('d-none');
        this.classList.add('d-none');
    });
}

// Exportar PDF
const btnPDF = document.getElementById('btnDownloadPDF');
if (btnPDF) {
    btnPDF.addEventListener('click', function () {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF('p', 'mm', 'a4');
        const diag = document.getElementById('txtDiagnostico').innerText;
        const conf = document.getElementById('txtConfianza').innerText;

        doc.setFont("helvetica", "bold");
        doc.setFontSize(18);
        doc.text("AgroAI - Reporte Fitosanitario de Palma de Aceite", 14, 20);
        
        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.text(`Fecha: ${new Date().toLocaleString()}`, 14, 28);
        
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text(`Diagnóstico: ${diag}`, 14, 42);
        doc.text(`${conf}`, 14, 49);

        const imgEntrada = document.getElementById('imgEntrante').src;
        const imgGradCAM = document.getElementById('imgGradCAM').src;
        const imgChart = document.getElementById('imgChart').src;

        if (imgEntrada && imgGradCAM) {
            doc.addImage(imgEntrada, 'PNG', 14, 60, 55, 55);
            doc.addImage(imgGradCAM, 'PNG', 75, 60, 55, 55);
            doc.addImage(imgChart, 'PNG', 136, 60, 60, 55);
        }

        doc.save(`Reporte_AgroAI_${diag.replace(/ /g, '_')}.pdf`);
    });
}