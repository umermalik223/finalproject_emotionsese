import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import cv2
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.orchestrator import EmotionSenseOrchestrator
from config import Config
from utils.logger import EmotionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Optional[EmotionSenseOrchestrator] = None
config = Config()

# FastAPI app for API interface
app = FastAPI(
    title="EmotionSense-AI",
    description="Advanced multimodal emotion analysis with therapeutic response generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global orchestrator
    
    logger.info("Starting EmotionSense-AI API Server...")
    
    # Initialize orchestrator in background thread to avoid blocking startup
    def init_orchestrator():
        global orchestrator
        try:
            logger.info("Loading AI models in background...")
            orchestrator = EmotionSenseOrchestrator(config.get_config())
            success = asyncio.run(orchestrator.initialize())
            if success:
                logger.info("✅ All AI models loaded successfully")
            else:
                logger.error("❌ Failed to load some AI models")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
    
    # Start model loading in background
    import threading
    threading.Thread(target=init_orchestrator, daemon=True).start()
    
    logger.info("✅ API Server started - Models loading in background")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    global orchestrator
    
    logger.info("Shutting down EmotionSense-AI System...")
    
    if orchestrator:
        await orchestrator.shutdown()
    
    logger.info("Shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if orchestrator and orchestrator.initialized:
        return {"status": "healthy", "system": "online", "models": "loaded"}
    else:
        return {"status": "partial", "system": "online", "models": "loading"}

@app.post("/analyze/multimodal")
async def analyze_multimodal(
    video_frame: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    """Analyze multimodal input (video frame + audio)"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Process video frame if provided
        video_frame_array = None
        if video_frame:
            # Read and decode video frame
            video_bytes = await video_frame.read()
            nparr = np.frombuffer(video_bytes, np.uint8)
            video_frame_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process audio file if provided
        audio_file_path = None
        if audio_file:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(await audio_file.read())
                audio_file_path = tmp_file.name
        
        # Process with orchestrator
        result = await orchestrator.process_multimodal_input(
            video_frame=video_frame_array,
            audio_file=audio_file_path,
            session_id=session_id
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Multimodal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/audio")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Analyze audio file only"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Read audio data
        audio_bytes = await audio_file.read()
        
        # Use librosa to properly handle different audio formats and convert to WAV
        audio_file_path = None
        audio_chunk = None
        
        try:
            import librosa
            import numpy as np
            import io
            import soundfile as sf
            import tempfile
            import subprocess
            
            logger.info(f"Processing audio file: {audio_file.content_type}, size: {len(audio_bytes)} bytes")
            
            # Try direct librosa loading first
            try:
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
                logger.info(f"Audio loaded directly: {len(audio_data)} samples at {sr}Hz")
            except Exception as librosa_error:
                logger.warning(f"Direct librosa failed: {librosa_error}")
                
                # Use FFmpeg for WebM/OGG conversion
                logger.info("Trying FFmpeg conversion for WebM/OGG format")
                
                # Save original bytes to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_input:
                    tmp_input.write(audio_bytes)
                    input_path = tmp_input.name
                
                # Convert with FFmpeg to WAV
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_output:
                    output_path = tmp_output.name
                
                # Use FFmpeg to convert
                ffmpeg_cmd = [
                    'ffmpeg', '-i', input_path, 
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',      # mono
                    '-f', 'wav',     # WAV format
                    output_path, '-y'
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Load converted audio
                    audio_data, sr = librosa.load(output_path, sr=16000, mono=True)
                    logger.info(f"Audio loaded via FFmpeg: {len(audio_data)} samples at {sr}Hz")
                    
                    # Cleanup temp files
                    import os
                    try:
                        os.unlink(input_path)
                        os.unlink(output_path)
                    except:
                        pass
                else:
                    logger.error(f"FFmpeg conversion failed: {result.stderr}")
                    raise Exception(f"FFmpeg failed: {result.stderr}")
            
            # Check audio quality
            if len(audio_data) > 0:
                audio_rms = np.sqrt(np.mean(audio_data**2))
                audio_max = np.max(np.abs(audio_data))
                logger.info(f"Audio quality - RMS: {audio_rms:.6f}, Max: {audio_max:.6f}")
                
                # Save as proper WAV file for speech-to-text
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    sf.write(tmp_file.name, audio_data, sr)
                    audio_file_path = tmp_file.name
                    logger.info(f"Audio saved as WAV: {audio_file_path}")
                
                # Prepare audio chunk for speech emotion analysis
                audio_chunk = audio_data
                logger.info(f"Audio prepared for speech emotion: {len(audio_chunk)} samples at {sr}Hz")
                
            else:
                logger.warning("No audio data loaded - possibly silent or corrupted file")
                
        except Exception as e:
            logger.error(f"Could not process audio: {e}")
            # Fallback: save raw bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                audio_file_path = tmp_file.name
            audio_chunk = None
        
        # Process with orchestrator (pass both audio_file and audio_chunk)
        result = await orchestrator.process_multimodal_input(
            audio_file=audio_file_path,
            audio_chunk=audio_chunk,
            session_id=session_id
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/video")
async def analyze_video(
    video_frame: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Analyze video frame only"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Read and decode video frame
        video_bytes = await video_frame.read()
        nparr = np.frombuffer(video_bytes, np.uint8)
        video_frame_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process with orchestrator
        result = await orchestrator.process_multimodal_input(
            video_frame=video_frame_array,
            session_id=session_id
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get session summary"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        summary = orchestrator.get_session_summary(session_id)
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Failed to get session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/text")
async def analyze_text(
    text: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Analyze text input only"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # For text-only analysis, we'll use a simplified approach
        # since the orchestrator expects multimodal input
        result = await orchestrator.analyze_text_only(text, session_id)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get system performance metrics"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        metrics = orchestrator.performance_profiler.export_performance_report()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_api_server():
    """Run FastAPI server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )

def run_web_ui():
    """Run Web UI with FastAPI"""
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    import os
    
    # Create FastAPI app for serving frontend
    frontend_app = FastAPI(
        title="EmotionSense Frontend",
        description="Web UI for EmotionSense AI",
        version="1.0.0"
    )
    
    # Add CORS middleware
    frontend_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Get the UI directory path
    ui_dir = os.path.join(os.path.dirname(__file__), 'ui')
    
    # Serve static files (CSS, JS)
    frontend_app.mount("/static", StaticFiles(directory=ui_dir), name="static")
    
    @frontend_app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_path = os.path.join(ui_dir, 'index.html')
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    @frontend_app.get("/style.css")
    async def serve_css():
        return FileResponse(os.path.join(ui_dir, 'style.css'), media_type='text/css')
    
    @frontend_app.get("/script.js")
    async def serve_js():
        return FileResponse(os.path.join(ui_dir, 'script.js'), media_type='application/javascript')
    
    # Handle any Streamlit legacy requests
    @frontend_app.get("/_stcore/{path:path}")
    async def handle_stcore_requests(path: str):
        return JSONResponse({"error": "Streamlit not available"}, status_code=404)
    
    @frontend_app.get("/healthz")
    async def health_check():
        return {"status": "healthy", "service": "frontend"}
    
    # Run the frontend server
    uvicorn.run(
        frontend_app, 
        host="0.0.0.0", 
        port=8501, 
        log_level="info",
        access_log=False  # Reduce noise
    )

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "api"  # Default mode
    
    logger.info(f"Starting EmotionSense-AI in {mode} mode")
    
    try:
        if mode == "api":
            logger.info("Starting API server on http://localhost:8003")
            run_api_server()
            
        elif mode == "ui":
            logger.info("Starting Web UI on http://localhost:8501")
            run_web_ui()
            
        elif mode == "both":
            import threading
            
            # Start API server in thread
            api_thread = threading.Thread(target=run_api_server, daemon=True)
            api_thread.start()
            
            logger.info("API server started on http://localhost:8003")
            logger.info("Starting Web UI on http://localhost:8501")
            
            # Start Web UI in main thread
            run_web_ui()
            
        else:
            logger.error(f"Unknown mode: {mode}")
            logger.info("Available modes: api, ui, both")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
