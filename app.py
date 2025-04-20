from flask import Flask, render_template, request, jsonify
from rag import initialize_components, load_and_process_document, create_rag_chain
import os
import subprocess
import sys

app = Flask(__name__)

# Initialize RAG components
embeddings, llm = initialize_components()
pdf_path = os.path.join(os.path.dirname(__file__), "Atlas-of-the-Heart-by-by-Bren-23.pdf")
texts = load_and_process_document(pdf_path)
rag_chain = create_rag_chain(embeddings, llm, texts)

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['opencv-python', 'numpy', 'transformers', 'Pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/chatbot')
def chatbot():
    return render_template('index.html')

@app.route('/emotion')
def emotion():
    return render_template('emotion.html')

@app.route('/start_emotion_detection', methods=['POST'])
def start_emotion_detection():
    try:
        # Check for required packages
        missing_packages = check_dependencies()
        if missing_packages:
            return jsonify({
                'status': 'error',
                'message': f'Missing required packages: {", ".join(missing_packages)}. Please install them first.'
            }), 400

        # Start emotion detection in a separate process
        python_executable = sys.executable
        emotion_script = os.path.join(os.path.dirname(__file__), 'emotion.py')
        
        subprocess.Popen([python_executable, emotion_script], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        return jsonify({
            'status': 'success',
            'message': 'Emotion detection started successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start emotion detection: {str(e)}'
        }), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Use the RAG chain to get the answer
        response = rag_chain.invoke({"input": question})
        answer = response['answer']
        
        return jsonify({
            'answer': answer,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
