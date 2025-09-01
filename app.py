
from flask import Flask, render_template_string, request, send_file, jsonify, Response, stream_with_context
import os
import time
import tempfile
import subprocess
import threading

app = Flask(__name__)


# --- SSE LOG STREAMING ENDPOINT (refactored) ---
import queue
log_queue = queue.Queue()
log_condition = threading.Condition()

@app.route('/stream_logs')
def stream_logs():
    def generate():
        # Stream log lines from the queue as they arrive
        while True:
            with log_condition:
                while log_queue.empty():
                    log_condition.wait()
                line = log_queue.get()
                if line is None:
                    break
                yield f"data: {line}\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# Global progress state
progress_state = {
    'progress': 0.0,
    'eta': '',
    'start_time': None,
    'last_update': None
}
progress_lock = threading.Lock()

@app.route('/progress')
def progress():
    with progress_lock:
        return jsonify({
            'progress': progress_state.get('progress', 0.0),
            'eta': progress_state.get('eta', '')
        })

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Eval UI - Brutalist</title>
    <style>
body {
    background: #fff;
    color: #000;
    font-family: 'Courier New', Courier, monospace;
    margin: 0;
    padding: 0;
    transition: background 0.2s, color 0.2s;
}
body.black-mode {
    background: #111;
    color: #fff;
}
.container {
    max-width: 600px;
    margin: 40px auto;
    background: #fff;
    border: 4px solid #000;
    box-shadow: 8px 8px 0 #000;
    padding: 32px 32px 24px 32px;
    transition: background 0.2s, color 0.2s;
}
body.black-mode .container {
    background: #222;
    border-color: #fff;
    box-shadow: 8px 8px 0 #fff;
}
h1 {
    font-size: 2.5rem;
    font-weight: bold;
    margin: 0 0 24px 0;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 4px solid #000;
    padding-bottom: 8px;
    background: yellow;
    display: inline-block;
    box-shadow: 4px 4px 0 #000;
    transition: background 0.2s, color 0.2s, border-bottom 0.2s, box-shadow 0.2s;
}
body.black-mode h1 {
    background: #fff;
    color: #111;
    border-bottom: 4px solid #fff;
    box-shadow: 4px 4px 0 #fff;
}
.mode-select {
    margin-bottom: 24px;
    font-size: 1.1rem;
    font-weight: bold;
}
label {
    background: #000;
    color: #fff;
    padding: 2px 8px;
    margin-right: 8px;
    font-size: 1rem;
    transition: background 0.2s, color 0.2s;
}
body.black-mode label {
    background: #fff;
    color: #111;
}
select {
    border: 2px solid #000;
    background: #fff;
    color: #000;
    font-size: 1rem;
    padding: 4px 8px;
    font-family: inherit;
    transition: background 0.2s, color 0.2s, border 0.2s;
}
body.black-mode select {
    border: 2px solid #fff;
    background: #222;
    color: #fff;
}
.drop-area {
    border: 4px solid #000;
    background: #eee;
    color: #000;
    padding: 48px 0;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 24px;
    cursor: pointer;
    box-shadow: 4px 4px 0 #000;
    text-transform: uppercase;
    transition: background 0.2s, color 0.2s, border 0.2s, box-shadow 0.2s;
}
.drop-area.dragover {
    background: #ff0;
    color: #000;
}
body.black-mode .drop-area {
    border: 4px solid #fff;
    background: #333;
    color: #fff;
    box-shadow: 4px 4px 0 #fff;
}
.upload-btn {
    background: #fff;
    color: #000;
    border: 4px solid #000;
    font-size: 1.1rem;
    font-weight: bold;
    padding: 12px 32px;
    cursor: pointer;
    box-shadow: 4px 4px 0 #000;
    text-transform: uppercase;
    margin-bottom: 16px;
    transition: background 0.2s, color 0.2s, border 0.2s, box-shadow 0.2s;
}
.upload-btn:hover {
    background: #000;
    color: #fff;
}
body.black-mode .upload-btn {
    background: #111;
    color: #fff;
    border: 4px solid #fff;
    box-shadow: 4px 4px 0 #fff;
}
body.black-mode .upload-btn:hover {
    background: #fff;
    color: #111;
}
#result {
    margin-top: 24px;
    font-size: 1.1rem;
    font-weight: bold;
    background: #ff0;
    border: 2px solid #000;
    color: #000;
    padding: 8px 12px;
    display: inline-block;
    box-shadow: 2px 2px 0 #000;
    transition: background 0.2s, color 0.2s, border 0.2s, box-shadow 0.2s;
}
body.black-mode #result {
    background: #fff;
    color: #111;
    border: 2px solid #fff;
    box-shadow: 2px 2px 0 #fff;
}
input[type="file"] {
    display: none;
}
    </style>
</head>
<body>
<div class="container">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <h1 style="margin-bottom: 0;">Eval UI</h1>
        <span id="toggle-mode" style="cursor:pointer; display:inline-block; width:40px; height:40px; vertical-align:middle;">
            <svg id="bulb-icon" width="40" height="40" viewBox="0 0 40 40" fill="orange" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="20" r="12" id="bulb-bulb" fill="orange" stroke="#000" stroke-width="3"/>
                <rect x="16" y="28" width="8" height="6" rx="2" id="bulb-base" fill="#888" stroke="#000" stroke-width="2"/>
            </svg>
        </span>
    </div>
    <form id="upload-form" enctype="multipart/form-data">
        <div style="margin-bottom: 18px;">
            <label for="title" style="display:block;margin-bottom:4px;">Title</label>
            <input id="title" name="title" type="text" style="width:100%;font-family:inherit;font-size:1rem;padding:6px;border:2px solid #000;box-sizing:border-box;" required placeholder="Enter a title for all rows..." />
        </div>
        <div style="margin-bottom: 18px;">
            <label for="jobdesc" style="display:block;margin-bottom:4px;">Job Description</label>
            <textarea id="jobdesc" name="jobdesc" rows="3" style="width:100%;font-family:inherit;font-size:1rem;padding:6px;border:2px solid #000;resize:vertical;box-sizing:border-box;" required placeholder="Paste the job description here..."></textarea>
        </div>
        <div style="margin-bottom: 18px;">
            <label for="recruitergpt" style="display:block;margin-bottom:4px;">Recruiter GPT</label>
            <textarea id="recruitergpt" name="recruitergpt" rows="3" style="width:100%;font-family:inherit;font-size:1rem;padding:6px;border:2px solid #000;resize:vertical;box-sizing:border-box;" required placeholder="Paste the Recruiter GPT response here..."></textarea>
        </div>
        <div class="mode-select">
            <label for="mode">Mode</label>
            <select id="mode" name="mode">
                <option value="maineval">maineval.py</option>
                <option value="candidatep">candidatep.py</option>
                <option value="template">template.py</option>
            </select>
        </div>
        <div class="drop-area" id="drop-area">
            Drag & Drop CSV file here or click to select
            <input type="file" id="fileElem" name="file" accept=".csv">
        </div>
        <button type="submit" class="upload-btn">Upload & Process</button>
    </form>
    <div id="result"></div>
    <pre id="log" style="background:#222;color:#fff;padding:12px;max-height:300px;overflow:auto;margin-top:20px;display:none;"></pre>
    <div id="progress-container" style="display:none; margin-top:20px;">
        <div style="width:100%;background:#eee;border:2px solid #000;height:32px;position:relative;">
            <div id="progress-bar" style="background:#ff0;height:100%;width:0%;transition:width 0.2s;"></div>
            <span id="progress-label" style="position:absolute;left:50%;top:0;transform:translateX(-50%);font-weight:bold;line-height:32px;color:#000;">0%</span>
        </div>
        <div id="eta" style="margin-top:8px;font-size:1rem;font-weight:bold;"></div>
    </div>
    <div id="download-section" style="display:none; margin-top:20px;">
        <a id="download-link" href="/download" class="upload-btn">Download CSV</a>
    </div>
</div>
<script>
    // Live log streaming
    let evtSource;
    function startLogStream() {
        const log = document.getElementById('log');
        log.style.display = 'block';
        log.textContent = '';
        evtSource = new EventSource('/stream_logs');
        evtSource.onmessage = function(e) {
            log.textContent += e.data;
            log.scrollTop = log.scrollHeight;
        };
        evtSource.onerror = function() {
            evtSource.close();
        };
    }
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('fileElem');
    let file;

    // Bulb mode toggle
    const toggleBtn = document.getElementById('toggle-mode');
    const bulbIcon = document.getElementById('bulb-icon');
    const bulbBulb = document.getElementById('bulb-bulb');
    let blackMode = false;
    function updateBulb() {
        if (blackMode) {
            bulbBulb.setAttribute('fill', '#111');
            bulbBulb.setAttribute('stroke', '#fff');
        } else {
            bulbBulb.setAttribute('fill', 'orange');
            bulbBulb.setAttribute('stroke', '#000');
        }
    }
    toggleBtn.addEventListener('click', () => {
        blackMode = !blackMode;
        document.body.classList.toggle('black-mode', blackMode);
        updateBulb();
    });
    updateBulb();

    dropArea.addEventListener('click', () => fileElem.click());
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('dragover');
    });
    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
        file = e.dataTransfer.files[0];
        dropArea.textContent = file.name;
    });
    fileElem.addEventListener('change', (e) => {
        file = e.target.files[0];
        dropArea.textContent = file.name;
    });
    document.getElementById('upload-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!file) {
            alert('Please select a CSV file.');
            return;
        }
        document.getElementById('progress-container').style.display = 'block';
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-label').textContent = '0%';
        document.getElementById('eta').textContent = '';
        document.getElementById('result').textContent = 'Processing...';
        document.getElementById('download-section').style.display = 'none';

    startLogStream();

        const formData = new FormData();
        formData.append('file', file);
    formData.append('mode', document.getElementById('mode').value);
    formData.append('title', document.getElementById('title').value);
    formData.append('jobdesc', document.getElementById('jobdesc').value);
    formData.append('recruitergpt', document.getElementById('recruitergpt').value);

        // Start polling for progress
        let polling = true;
        let startTime = Date.now();
        function pollProgress() {
            if (!polling) return;
            fetch('/progress').then(r => r.json()).then(data => {
                let percent = Math.round((data.progress || 0) * 100);
                document.getElementById('progress-bar').style.width = percent + '%';
                document.getElementById('progress-label').textContent = percent + '%';
                if (data.eta) {
                    document.getElementById('eta').textContent = 'Estimated time left: ' + data.eta;
                }
                if (percent < 100) {
                    setTimeout(pollProgress, 1000);
                }
            }).catch(() => setTimeout(pollProgress, 2000));
        }
        pollProgress();

        const res = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        polling = false;
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-label').textContent = '100%';
        document.getElementById('eta').textContent = '';
    if (evtSource) evtSource.close();
        if (res.ok) {
            document.getElementById('result').textContent = 'Processing complete!';
            document.getElementById('download-section').style.display = 'block';
            document.getElementById('log').style.display = 'block';
        } else {
            document.getElementById('result').textContent = 'Processing failed.';
            document.getElementById('download-section').style.display = 'none';
            document.getElementById('log').style.display = 'block';
        }
    });
</script>
</body>
</html>
'''
    # Removed all remaining indented or stray HTML code after the flush-left HTML string. Only the HTML variable at the top remains.
    # Removed duplicate and broken HTML assignments below. Only the flush-left HTML string above remains.

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/process', methods=['POST'])
def process():
    import pandas as pd
    import io
    file = request.files['file']
    mode = request.form['mode']
    title = request.form.get('title', '')
    jobdesc = request.form.get('jobdesc', '')
    recruitergpt = request.form.get('recruitergpt', '')
    # Save to persistent results directory
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    input_path = os.path.join(results_dir, 'input.csv')
    output_path = os.path.join(results_dir, 'processed.csv')
    df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
    df['Grapevine Job - Job → Description'] = jobdesc
    df['Recruiter GPT Response '] = recruitergpt
    df.to_csv(input_path, index=False)
    # Remove old processed.csv if it exists, to avoid confusion
    if os.path.exists(output_path):
        os.remove(output_path)
    script_map = {
        'maineval': 'maineval.py',
        'candidatep': 'candidatep.py',
        'template': 'template.py',
    }
    script = script_map.get(mode, 'maineval.py')
    import traceback
    try:
        # Start progress
        with progress_lock:
            progress_state['progress'] = 0.01
            progress_state['start_time'] = time.time()
            progress_state['last_update'] = time.time()
            progress_state['eta'] = ''

        def run_and_track():
            try:
                # Actually run the script, passing title as third argument
                command = [os.path.join('ven', 'Scripts', 'python.exe'), script, input_path, output_path, title]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
                if process.stdout is not None:
                    for line in iter(process.stdout.readline, ''):
                        with log_condition:
                            log_queue.put(line)
                            log_condition.notify_all()
                    process.stdout.close()
                process.wait()
                with progress_lock:
                    progress_state['progress'] = 1.0
                    progress_state['eta'] = ''
            except subprocess.CalledProcessError as e:
                with progress_lock:
                    progress_state['progress'] = 1.0
                    progress_state['eta'] = ''
                with log_condition:
                    log_queue.put(f'PROCESSING ERROR:\nReturn code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}\n')
                    log_condition.notify_all()
            except Exception as ex:
                tb = traceback.format_exc()
                with progress_lock:
                    progress_state['progress'] = 1.0
                    progress_state['eta'] = ''
                with log_condition:
                    log_queue.put(f'UNEXPECTED ERROR: {ex}\n{tb}\n')
                    log_condition.notify_all()
            finally:
                with log_condition:
                    log_queue.put(None)  # Signal end of log stream
                    log_condition.notify_all()

        t = threading.Thread(target=run_and_track)
        t.start()
        t.join()
        # After thread completes, check if processed.csv was created
        if not os.path.exists(output_path):
            with log_condition:
                log_queue.put('ERROR: processed.csv was not created by the script.\n')
                log_queue.put(None)
                log_condition.notify_all()
            return "Processing failed: processed.csv was not created.", 500
        return jsonify({'success': True})
    except Exception as ex:
        import traceback
        tb = traceback.format_exc()
        with progress_lock:
            progress_state['progress'] = 1.0
            progress_state['eta'] = ''
        with log_condition:
            log_queue.put(f'UNEXPECTED ERROR: {ex}\n{tb}\n')
            log_queue.put(None)
            log_condition.notify_all()
        return f"Unexpected error: {str(ex)}", 500

@app.route('/download')
def download():
    results_dir = os.path.join(os.getcwd(), 'results')
    output_path = os.path.join(results_dir, 'processed.csv')
    if not os.path.exists(output_path):
        return "No processed file found. Please process a file first.", 404
    return send_file(output_path, as_attachment=True, download_name='processed.csv')

if __name__ == '__main__':
    app.run(debug=True)
