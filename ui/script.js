// EmotionSense AI Frontend JavaScript

class EmotionSenseApp {
    constructor() {
        this.isSessionActive = false;
        this.mediaStream = null; // For audio recording
        this.cameraStream = null; // For video camera
        this.mediaRecorder = null;
        this.isRecording = false;
        this.sessionId = null;
        this.videoElement = null;
        this.canvasElement = null;
        this.analysisInterval = null;
        
        // API Configuration
        this.apiBaseUrl = 'http://localhost:8003';
        
        // Emotion emojis mapping - updated for actual model outputs
        this.emotionEmojis = {
            // Basic emotions
            happy: '😊', joy: '😄', excited: '🤩',
            sad: '😢', crying: '😭', depressed: '😞',
            angry: '😠', furious: '😡', annoyed: '😤',
            fear: '😨', scared: '😰', anxious: '😟',
            surprise: '😲', shocked: '😱', amazed: '😯',
            disgust: '🤢', sick: '🤮', disgusted: '😖',
            neutral: '😐', calm: '😌', peaceful: '☺️',
            confused: '😕', uncertain: '🤔', puzzled: '😵',
            
            // Model-specific outputs (your actual model responses)
            'positive/happy': '😊',
            'negative/sad': '😢', 
            'normal/stable': '😌',
            'anger/irritation': '😠',
            'fear/anxiety': '😟',
            'disgust/aversion': '🤢',
            'joy/happiness': '😄',
            'sadness/melancholy': '😢',
            'surprise/shock': '😲',
            
            // Fallback
            positive: '😊',
            negative: '😞',
            normal: '😐',
            stable: '😌'
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkAPIStatus();
        this.generateSessionId();
    }
    
    setupEventListeners() {
        // Session button
        document.getElementById('sessionBtn').addEventListener('click', () => {
            this.toggleSession();
        });

        // Chat input
        document.getElementById('sendBtn').addEventListener('click', () => {
            this.sendChatMessage();
        });
        
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });
        
        // Checkbox listeners for model exclusions
        ['excludeFacial', 'excludeSpeech', 'excludeText'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                this.updateModelSettings();
            });
        });
    }
    
    generateSessionId() {
        this.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        console.log('Generated session ID:', this.sessionId);
    }
    
    async checkAPIStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            const statusElement = document.getElementById('apiStatus');
            const statusDot = statusElement.querySelector('.status-dot');
            
            if (data.status === 'healthy') {
                statusDot.className = 'status-dot online';
                statusElement.textContent = 'API: Connected';
            } else {
                statusDot.className = 'status-dot loading';
                statusElement.textContent = 'API: Models Loading...';
            }
        } catch (error) {
            console.error('API health check failed:', error);
            const statusElement = document.getElementById('apiStatus');
            const statusDot = statusElement.querySelector('.status-dot');
            statusDot.className = 'status-dot offline';
            statusElement.textContent = 'API: Offline';
        }
    }
    
    async toggleSession() {
        const sessionBtn = document.getElementById('sessionBtn');
        
        if (!this.isSessionActive) {
            await this.startSession();
            sessionBtn.textContent = 'Stop Session';
            sessionBtn.style.background = 'linear-gradient(90deg, #ff6b6b, #ff8e8e)';
        } else {
            await this.stopSession();
            sessionBtn.textContent = 'Start Session';
            sessionBtn.style.background = 'linear-gradient(90deg, #43e97b, #38f9d7)';
        }
    }
    
    async startSession() {
        try {
            // Clear previous messages
            document.getElementById('chatMessages').innerHTML = '';
            
            // Request camera and microphone permissions
            await this.setupCamera();
            await this.setupMicrophone();
            
            this.isSessionActive = true;
            this.isRecording = true; // Auto-start recording with session
            
            // Start continuous analysis and recording
            this.startContinuousAnalysis();
            await this.startRecording();
            
        } catch (error) {
            console.error('Failed to start session:', error);
            this.addChatMessage('speech', 'Failed to start session. Please check camera and microphone permissions.');
        }
    }
    
    async stopSession() {
        this.isSessionActive = false;
        console.log('🛑 Stopping session and all media streams...');
        
        // Stop continuous analysis
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
            this.analysisInterval = null;
        }
        
        // Stop recording if active
        if (this.isRecording) {
            this.stopRecording();
        }
        
        // Stop audio media stream (microphone)
        if (this.mediaStream) {
            console.log('🎤 Stopping audio stream...');
            this.mediaStream.getTracks().forEach(track => {
                console.log(`🎤 Stopping audio track: ${track.kind} - ${track.label}`);
                track.stop();
            });
            this.mediaStream = null;
        }
        
        // Stop camera stream (video)
        if (this.cameraStream) {
            console.log('📹 Stopping camera stream...');
            this.cameraStream.getTracks().forEach(track => {
                console.log(`📹 Stopping camera track: ${track.kind} - ${track.label}`);
                track.stop();
            });
            this.cameraStream = null;
        }
        
        // Clear video element source to ensure camera is fully released
        if (this.videoElement) {
            console.log('📹 Clearing video element source...');
            this.videoElement.srcObject = null;
        }
        
        // Hide video container
        document.getElementById('videoContainer').style.display = 'none';
        
        // Update status indicators
        this.updateStatusIndicator('cameraStatus', 'offline', 'Camera: Offline');
        this.updateStatusIndicator('micStatus', 'offline', 'Microphone: Offline');
        
        // Reset emotion cards
        this.resetEmotionCards();
        
        // Clear chat messages
        document.getElementById('chatMessages').innerHTML = '<p class="speech-text">Speech-to-text will appear here when you start a session...</p>';
        
        console.log('✅ Session stopped - all media streams released');
    }
    
    async setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 }, 
                audio: false 
            });
            
            // Store the camera stream separately
            this.cameraStream = stream;
            
            this.videoElement = document.getElementById('videoPreview');
            this.canvasElement = document.getElementById('captureCanvas');
            
            this.videoElement.srcObject = stream;
            
            // Show video container
            document.getElementById('videoContainer').style.display = 'block';
            
            this.updateStatusIndicator('cameraStatus', 'online', 'Camera: Active');
            
            console.log('📹 Camera stream started and stored');
            return stream;
        } catch (error) {
            console.error('Camera setup failed:', error);
            this.updateStatusIndicator('cameraStatus', 'offline', 'Camera: Failed');
            throw error;
        }
    }
    
    async setupMicrophone() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 48000,  // Higher sample rate for better quality
                    channelCount: 1,     // Mono
                    echoCancellation: true,
                    noiseSuppression: false,  // Keep speech natural
                    autoGainControl: true,   // Help with volume
                    volume: 1.0,
                    // Additional constraints for better speech capture
                    googEchoCancellation: true,
                    googAutoGainControl: true,
                    googNoiseSuppression: false,
                    googHighpassFilter: false,
                    googDucking: false
                }, 
                video: false 
            });
            
            this.mediaStream = stream;
            this.updateStatusIndicator('micStatus', 'online', 'Microphone: Ready');
            
            return stream;
        } catch (error) {
            console.error('Microphone setup failed:', error);
            this.updateStatusIndicator('micStatus', 'offline', 'Microphone: Failed');
            throw error;
        }
    }
    
    startContinuousAnalysis() {
        // Analyze every 2 seconds when session is active
        this.analysisInterval = setInterval(() => {
            if (this.isSessionActive) {
                this.captureAndAnalyze();
            }
        }, 2000);
    }
    
    async captureAndAnalyze() {
        if (!this.isSessionActive || !this.videoElement) return;
        
        try {
            // Capture current video frame
            const canvas = this.canvasElement;
            const ctx = canvas.getContext('2d');
            
            canvas.width = this.videoElement.videoWidth;
            canvas.height = this.videoElement.videoHeight;
            
            ctx.drawImage(this.videoElement, 0, 0);
            
            // Convert to blob
            canvas.toBlob(async (blob) => {
                if (blob) {
                    await this.analyzeVideoFrame(blob);
                }
            }, 'image/jpeg', 0.8);
            
        } catch (error) {
            console.error('Capture and analyze failed:', error);
        }
    }
    
    async analyzeVideoFrame(videoBlob) {
        try {
            const formData = new FormData();
            formData.append('video_frame', videoBlob, 'frame.jpg');
            formData.append('session_id', this.sessionId);
            
            const response = await fetch(`${this.apiBaseUrl}/analyze/video`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.updateEmotionDisplay(result);
            }
        } catch (error) {
            console.error('Video analysis failed:', error);
        }
    }
    
    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            this.stopRecording();
        }
    }
    
    async startRecording() {
        if (!this.mediaStream) {
            return;
        }
        
        try {
            // Try different mime types for better compatibility - prioritize WAV and OGG over WebM
            let mimeType = 'audio/wav';
            if (MediaRecorder.isTypeSupported('audio/wav')) {
                mimeType = 'audio/wav';
            } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                mimeType = 'audio/ogg;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                mimeType = 'audio/ogg';
            } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                mimeType = 'audio/webm';
            } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                mimeType = 'audio/webm;codecs=opus';
            }
            
            console.log(`🎤 Selected audio format: ${mimeType}`);
            
            this.mediaRecorder = new MediaRecorder(this.mediaStream, {
                mimeType: mimeType
            });
            const audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = async () => {
                if (audioChunks.length > 0) {
                    const audioBlob = new Blob(audioChunks, { type: mimeType });
                    console.log(`🎤 Audio blob created: ${audioBlob.size} bytes, type: ${mimeType}`);
                    
                    // Debug: Check audio quality before sending
                    if (audioBlob.size < 1000) {
                        console.warn(`⚠️ Audio blob seems very small (${audioBlob.size} bytes) - might be silence`);
                    } else {
                        console.log(`✅ Audio blob looks good (${audioBlob.size} bytes)`);
                    }
                    
                    await this.analyzeAudio(audioBlob);
                } else {
                    console.warn(`⚠️ No audio chunks recorded`);
                }
                
                // Auto-restart recording if session is still active
                if (this.isSessionActive) {
                    setTimeout(() => {
                        this.startRecording();
                    }, 500);
                }
            };
            
            // Record in 3-second chunks for better responsiveness while maintaining quality
            this.mediaRecorder.start(3000);
            this.isRecording = true;
            console.log(`Started recording with ${mimeType}`);
            
        } catch (error) {
            console.error('Recording failed:', error);
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
        }
    }
    
    async analyzeAudio(audioBlob) {
        try {
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'recording.wav');
            formData.append('session_id', this.sessionId);
            
            const response = await fetch(`${this.apiBaseUrl}/analyze/audio`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.updateEmotionDisplay(result);
                
                // Display speech-to-text result with better debugging
                console.log('🎤 Checking speech-to-text result:', result.speech_to_text);
                if (result.speech_to_text) {
                    // Get text and ensure it's not empty or whitespace only
                    let text = result.speech_to_text.text || '';
                    text = text.trim();
                    
                    // Check confidence and quality
                    const confidence = result.speech_to_text.confidence || 0;
                    const hasError = result.speech_to_text.error;
                    
                    // More intelligent handling
                    if (!text || text.length < 2) {
                        if (hasError) {
                            console.log('⚠️ Speech-to-text error:', hasError);
                            text = '[Audio too quiet or short]';
                        } else if (confidence < 0.3) {
                            console.log('⚠️ Low confidence transcription:', confidence);
                            text = '[Unclear speech - try speaking louder]';
                        } else {
                            text = '[Brief sound detected]';
                        }
                    } else {
                        // Valid transcription - check for very short words that might be noise
                        if (text.length < 4 && confidence < 0.5) {
                            text = `"${text}" (uncertain - try speaking more clearly)`;
                        }
                    }
                    
                    console.log('✅ Displaying speech-to-text:', text);
                    this.addChatMessage('speech', text);
                } else {
                    console.log('⚠️ No speech_to_text found in result');
                    this.addChatMessage('speech', '[No audio processed]');
                }
                
                // Display therapeutic response if available
                console.log('🔍 Checking for therapeutic_response in result:', !!result.therapeutic_response);
                if (result.therapeutic_response) {
                    console.log('💙 Therapeutic response structure:', Object.keys(result.therapeutic_response));
                    console.log('💙 Full therapeutic response:', result.therapeutic_response);
                    
                    if (result.therapeutic_response.therapeutic_response) {
                        console.log('💙 Displaying therapeutic response (nested structure)');
                        this.displayTherapeuticResponse(result.therapeutic_response.therapeutic_response);
                    } else if (result.therapeutic_response.success) {
                        console.log('💙 Displaying therapeutic response (direct structure)');
                        this.displayTherapeuticResponse(result.therapeutic_response);
                    } else {
                        console.log('⚠️ Therapeutic response found but no valid structure');
                    }
                } else {
                    console.log('⚠️ No therapeutic_response found in result');
                }
            }
        } catch (error) {
            console.error('Audio analysis failed:', error);
            this.addChatMessage('system', 'Audio analysis failed.');
        }
    }
    
    updateEmotionDisplay(result) {
        console.log('🎭 Updating emotion display with result:', result);
        
        // Update facial emotion
        if (result.facial_emotion) {
            console.log('😊 Updating facial emotion:', result.facial_emotion);
            this.updateEmotionCard('facialEmotion', 'facialConfidence', result.facial_emotion);
        } else {
            console.log('❌ No facial_emotion in result');
        }
        
        // Update speech emotion
        if (result.speech_emotion) {
            console.log('🔊 Updating speech emotion:', result.speech_emotion);
            this.updateEmotionCard('speechEmotion', 'speechConfidence', result.speech_emotion);
        } else {
            console.log('❌ No speech_emotion in result');
        }
        
        // Update text emotion
        if (result.text_emotion) {
            console.log('📝 Updating text emotion:', result.text_emotion);
            this.updateEmotionCard('textEmotion', 'textConfidence', result.text_emotion);
        } else {
            console.log('❌ No text_emotion in result');
        }
        
        // Update mental state
        if (result.mental_state) {
            console.log('🧠 Updating mental state:', result.mental_state);
            this.updateEmotionCard('mentalState', 'mentalConfidence', result.mental_state);
        } else {
            console.log('❌ No mental_state in result');
        }
        
        // Update fused emotion
        if (result.fused_emotion) {
            console.log('🔀 Updating fused emotion:', result.fused_emotion);
            this.updateEmotionCard('fusedEmotion', 'fusedConfidence', result.fused_emotion);
        } else {
            console.log('❌ No fused_emotion in result');
        }
        
        // Add glow effect to indicate update
        document.querySelectorAll('.card').forEach(card => {
            card.classList.add('updating');
            setTimeout(() => card.classList.remove('updating'), 1500);
        });
    }
    
    updateEmotionCard(emotionElementId, confidenceElementId, emotionData) {
        console.log(`🔄 Updating ${emotionElementId} with data:`, emotionData);
        
        const emotionElement = document.getElementById(emotionElementId);
        const confidenceElement = document.getElementById(confidenceElementId);
        
        if (!emotionElement || !confidenceElement) {
            console.error(`❌ Elements not found: ${emotionElementId}, ${confidenceElementId}`);
            return;
        }
        
        if (emotionData && emotionData.emotion) {
            const originalEmotion = emotionData.emotion;
            const emotion = originalEmotion.toLowerCase();
            const emoji = this.emotionEmojis[emotion] || this.emotionEmojis[emotion.split('/')[0]] || '🤔';
            const confidence = emotionData.confidence ? Math.round(emotionData.confidence * 100) : 0;
            
            const displayEmotion = this.formatEmotionForDisplay(originalEmotion);
            
            emotionElement.textContent = `${emoji} ${displayEmotion}`;
            confidenceElement.textContent = `${confidence}%`;
            
            console.log(`✅ Successfully updated ${emotionElementId}: ${emoji} ${displayEmotion} (${confidence}%)`);
        } else {
            console.log(`⚠️ No valid emotion data for ${emotionElementId}:`, emotionData);
        }
    }
    
    formatEmotionForDisplay(emotion) {
        // Handle format like "Positive/Happy" -> "Happy" 
        if (emotion.includes('/')) {
            const parts = emotion.split('/');
            return this.capitalizeFirst(parts[1] || parts[0]);
        }
        return this.capitalizeFirst(emotion);
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
    
    sendChatMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (message) {
            this.addChatMessage('user', message);
            input.value = '';
            
            // Analyze text emotion
            this.analyzeTextEmotion(message);
        }
    }
    
    async analyzeTextEmotion(text) {
        try {
            const formData = new FormData();
            formData.append('text', text);
            formData.append('session_id', this.sessionId);
            
            const response = await fetch(`${this.apiBaseUrl}/analyze/text`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.updateEmotionDisplay(result);
                
                // Display therapeutic response if available (text analysis can also trigger therapeutic responses)
                console.log('🔍 Checking for therapeutic_response in text result:', !!result.therapeutic_response);
                if (result.therapeutic_response) {
                    console.log('💙 Therapeutic response structure:', Object.keys(result.therapeutic_response));
                    console.log('💙 Full therapeutic response:', result.therapeutic_response);
                    
                    if (result.therapeutic_response.therapeutic_response) {
                        console.log('💙 Displaying therapeutic response (nested structure)');
                        this.displayTherapeuticResponse(result.therapeutic_response.therapeutic_response);
                    } else if (result.therapeutic_response.success) {
                        console.log('💙 Displaying therapeutic response (direct structure)');
                        this.displayTherapeuticResponse(result.therapeutic_response);
                    } else {
                        console.log('⚠️ Therapeutic response found but no valid structure');
                    }
                } else {
                    console.log('⚠️ No therapeutic_response found in text result');
                }
            } else {
                console.error('Text analysis failed:', response.statusText);
            }
            
        } catch (error) {
            console.error('Text analysis failed:', error);
        }
    }
    
    addChatMessage(type, message) {
        console.log(`📧 addChatMessage called with type: '${type}', message length: ${message ? message.length : 0}`);
        
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            console.error('❌ chatMessages element not found!');
            return;
        }
        
        const messageElement = document.createElement('p');
        
        if (type === 'speech') {
            messageElement.className = 'speech-text';
            messageElement.textContent = message; // No prefix for speech-to-text
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            console.log('✅ Speech message added to DOM');
        } else if (type === 'user') {
            messageElement.className = 'user-message';
            messageElement.innerHTML = `<strong>You:</strong> ${message}`;
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            console.log('✅ User message added to DOM');
        } else if (type === 'therapeutic') {
            messageElement.className = 'therapeutic-message';
            messageElement.innerHTML = message;
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            console.log('✅ Therapeutic message added to DOM');
            console.log('📊 Total chat messages now:', chatMessages.children.length);
        } else {
            console.log(`⚠️ Ignoring message type: '${type}'`);
        }
        // Ignore system messages and other types
    }
    
    displayTherapeuticResponse(therapeuticData) {
        console.log('💙 Formatting therapeutic response:', therapeuticData);
        console.log('💙 Type of therapeuticData:', typeof therapeuticData);
        console.log('💙 Keys in therapeuticData:', Object.keys(therapeuticData || {}));
        
        // Check if we have the data
        if (!therapeuticData) {
            console.error('❌ No therapeutic data provided');
            return;
        }
        
        // Create comprehensive therapeutic response display with NEW 5-step structure
        let responseHtml = `
            <div class="therapeutic-response">
                <div class="therapeutic-header">💙 <strong>AI Therapeutic Assistant</strong></div>
                
                <div class="situation-acknowledgment">
                    <strong>💙 Step 1 - Understanding & Support:</strong> ${therapeuticData.acknowledgment_and_support || therapeuticData.situation_acknowledgment || therapeuticData.empathetic_response || 'I\'m here to support you.'}
                </div>
                
                ${therapeuticData.proven_calming_techniques && therapeuticData.proven_calming_techniques.length > 0 ? `
                <div class="evidence-techniques">
                    <strong>🧠 Step 2 - Proven Calming Techniques:</strong>
                    <ol>
                        ${therapeuticData.proven_calming_techniques.map(technique => `<li>${technique}</li>`).join('')}
                    </ol>
                </div>
                ` : therapeuticData.evidence_based_techniques && therapeuticData.evidence_based_techniques.length > 0 ? `
                <div class="evidence-techniques">
                    <strong>🧠 Step 2 - Evidence-Based Techniques:</strong>
                    <ol>
                        ${therapeuticData.evidence_based_techniques.map(technique => `<li>${technique}</li>`).join('')}
                    </ol>
                </div>
                ` : therapeuticData.calming_techniques && therapeuticData.calming_techniques.length > 0 ? `
                <div class="evidence-techniques">
                    <strong>🧠 Step 2 - Calming Techniques:</strong>
                    <ol>
                        ${therapeuticData.calming_techniques.map(technique => `<li>${technique}</li>`).join('')}
                    </ol>
                </div>
                ` : ''}
                
                ${therapeuticData.after_calming_suggestions ? `
                <div class="practical-suggestions">
                    <strong>🎯 Step 3 - After Calming Suggestions:</strong> ${therapeuticData.after_calming_suggestions}
                </div>
                ` : therapeuticData.practical_suggestions ? `
                <div class="practical-suggestions">
                    <strong>🎯 Step 3 - Practical Suggestions:</strong> ${therapeuticData.practical_suggestions}
                </div>
                ` : therapeuticData.personalized_guidance ? `
                <div class="practical-suggestions">
                    <strong>🎯 Step 3 - Personalized Guidance:</strong> ${therapeuticData.personalized_guidance}
                </div>
                ` : therapeuticData.emotional_validation ? `
                <div class="practical-suggestions">
                    <strong>✨ Step 3 - Validation:</strong> ${therapeuticData.emotional_validation}
                </div>
                ` : ''}
                
                ${therapeuticData.optional_additional_help && therapeuticData.optional_additional_help !== 'Not applicable for this situation.' ? `
                <div class="additional-help">
                    <strong>📋 Step 4 - Additional Help:</strong> ${therapeuticData.optional_additional_help}
                </div>
                ` : ''}
                
                ${therapeuticData.system_analysis ? `
                <div class="system-analysis">
                    <strong>📊 Step 5 - How We Analyzed Your Emotions:</strong>
                    <ul>
                        <li><strong>Analysis Method:</strong> ${therapeuticData.system_analysis.how_we_analyzed || 'Multimodal emotion detection'}</li>
                        <li><strong>Confidence Levels:</strong> ${therapeuticData.system_analysis.confidence_levels || 'Not available'}</li>
                        <li><strong>Primary Indicators:</strong> ${therapeuticData.system_analysis.primary_indicators || 'Not available'}</li>
                    </ul>
                </div>
                ` : therapeuticData.multimodal_analysis ? `
                <div class="system-analysis">
                    <strong>📊 Step 5 - Multimodal Analysis:</strong>
                    <ul>
                        <li><strong>Primary Emotion:</strong> ${therapeuticData.multimodal_analysis.primary_emotion || 'Unknown'}</li>
                        <li><strong>Confidence:</strong> ${therapeuticData.multimodal_analysis.confidence_summary || 'Not available'}</li>
                        <li><strong>Coherence:</strong> ${therapeuticData.multimodal_analysis.coherence_note || 'Not available'}</li>
                    </ul>
                </div>
                ` : therapeuticData.emotion_insights ? `
                <div class="system-analysis">
                    <strong>📊 Step 5 - Emotion Analysis:</strong>
                    <ul>
                        <li><strong>Primary Emotion:</strong> ${therapeuticData.emotion_insights.primary_emotion || 'Unknown'}</li>
                        <li><strong>Confidence:</strong> ${therapeuticData.emotion_insights.confidence_summary || 'Not available'}</li>
                        <li><strong>Coherence:</strong> ${therapeuticData.emotion_insights.coherence_note || 'Not available'}</li>
                    </ul>
                </div>
                ` : ''}
                
                <div class="severity-indicator severity-${therapeuticData.severity_assessment || 'low'}">
                    <strong>Assessment Level:</strong> ${(therapeuticData.severity_assessment || 'low').toUpperCase()}
                </div>
            </div>
        `;
        
        console.log('💙 Generated HTML length:', responseHtml.length);
        console.log('💙 About to call addChatMessage with therapeutic type');
        
        this.addChatMessage('therapeutic', responseHtml);
        
        console.log('💙 addChatMessage called successfully');
    }
    
    updateStatusIndicator(elementId, status, text) {
        const element = document.getElementById(elementId);
        const dot = element.querySelector('.status-dot');
        
        dot.className = `status-dot ${status}`;
        element.innerHTML = `<span class="status-dot ${status}"></span>${text}`;
    }
    
    resetEmotionCards() {
        this.updateSimpleEmotionCard('facialEmotion', 'facialConfidence', '😐 Waiting...', '0%');
        this.updateSimpleEmotionCard('speechEmotion', 'speechConfidence', '🔇 Waiting...', '0%');
        this.updateSimpleEmotionCard('textEmotion', 'textConfidence', '📝 Waiting...', '0%');
        this.updateSimpleEmotionCard('mentalState', 'mentalConfidence', '🧠 Analyzing...', '0%');
        this.updateSimpleEmotionCard('fusedEmotion', 'fusedConfidence', '🔀 Overall: Analyzing...', '0%');
    }
    
    updateSimpleEmotionCard(emotionElementId, confidenceElementId, emotionText, confidenceText) {
        document.getElementById(emotionElementId).textContent = emotionText;
        document.getElementById(confidenceElementId).textContent = confidenceText;
    }
    
    updateModelSettings() {
        // Get current checkbox states
        const excludeFacial = document.getElementById('excludeFacial').checked;
        const excludeSpeech = document.getElementById('excludeSpeech').checked;
        const excludeText = document.getElementById('excludeText').checked;
        
        // Update card visibility or styling based on excluded models
        document.getElementById('facialEmotionCard').style.opacity = excludeFacial ? '0.5' : '1';
        document.getElementById('speechEmotionCard').style.opacity = excludeSpeech ? '0.5' : '1';
        document.getElementById('textEmotionCard').style.opacity = excludeText ? '0.5' : '1';
        
        console.log('Model settings updated:', {
            facial: !excludeFacial,
            speech: !excludeSpeech,
            text: !excludeText
        });
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.emotionSenseApp = new EmotionSenseApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.emotionSenseApp) {
        // Pause analysis when tab is not visible to save resources
        if (window.emotionSenseApp.analysisInterval) {
            clearInterval(window.emotionSenseApp.analysisInterval);
        }
    } else if (window.emotionSenseApp && window.emotionSenseApp.isSessionActive) {
        // Resume analysis when tab becomes visible
        window.emotionSenseApp.startContinuousAnalysis();
    }
});

// Handle page unload to ensure camera is stopped
window.addEventListener('beforeunload', () => {
    if (window.emotionSenseApp && window.emotionSenseApp.isSessionActive) {
        console.log('🛑 Page unloading - stopping all media streams...');
        window.emotionSenseApp.stopSession();
    }
});

// Handle page focus loss (additional safety)
window.addEventListener('blur', () => {
    if (window.emotionSenseApp && window.emotionSenseApp.isSessionActive) {
        console.log('🛑 Window lost focus - ensuring media streams are properly managed...');
        // Don't stop completely, but ensure streams are tracked
    }
});