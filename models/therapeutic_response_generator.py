import openai
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import aiohttp
import time

logger = logging.getLogger(__name__)

class TherapeuticResponseGenerator:
     def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.response_cache = {}
        self.max_retries = 3
        
    async def generate_therapeutic_response(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate empathetic therapeutic response with practical calming techniques"""
        try:
            # Format multimodal data for GPT-4
            formatted_prompt = self._format_user_data_for_gpt4(user_data)
            
            # Generate response with retry logic
            response = await self._call_gpt4_with_retries(formatted_prompt)
            
            # Parse and validate response
            therapeutic_response = self._parse_gpt4_response(response)
            
            return {
                'therapeutic_response': therapeutic_response,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"🚨 Therapeutic response generation FAILED - NO FALLBACKS ALLOWED: {e}")
            raise Exception(f"Therapeutic response generation failed: {e}")
    
    def _format_user_data_for_gpt4(self, user_data: Dict[str, Any]) -> str:
        """Format extracted multimodal data for GPT-4o-mini therapeutic analysis"""
        
        # Extract key emotional indicators
        face_emotion = user_data.get('face_emotion', {})
        speech_emotion = user_data.get('speech_emotion', {})
        text_emotion = user_data.get('text_emotion', {})
        mental_state = user_data.get('mental_state', {})
        emotion_fusion = user_data.get('emotion_fusion', {})
        transcribed_text = user_data.get('transcribed_text', '')
        
        # Format confidence percentages
        face_conf = round(face_emotion.get('confidence', 0) * 100, 1)
        speech_conf = round(speech_emotion.get('confidence', 0) * 100, 1)
        text_conf = round(text_emotion.get('confidence', 0) * 100, 1)
        mental_conf = round(mental_state.get('confidence', 0) * 100, 1)
        fusion_conf = round(emotion_fusion.get('confidence', 0) * 100, 1)
        
        prompt = f"""You are an empathetic AI therapeutic assistant. Based on the user's emotional state and words, provide a personalized therapeutic response following this EXACT STRUCTURE:

EMOTIONAL ANALYSIS:
- Facial Expression: {face_emotion.get('emotion', 'neutral')} ({face_conf}% confidence)
- Voice Tone: {speech_emotion.get('emotion', 'neutral')} ({speech_conf}% confidence)  
- Text Sentiment: {text_emotion.get('emotion', 'neutral')} ({text_conf}% confidence)
- Mental State: {mental_state.get('mental_state', 'stable')} ({mental_conf}% confidence)
- Overall Emotion: {emotion_fusion.get('fused_emotion', 'neutral')} ({fusion_conf}% confidence)

USER'S WORDS: "{transcribed_text}"

Respond in this EXACT JSON structure:
{{
    "acknowledgment_and_support": "Acknowledge that their feelings are NORMAL among people. Give them confidence they are NOT ALONE. Say 'I am with you' or similar. Show understanding of their specific situation. If emotion is severe (high stress/anxiety/depression), mention: 'Please consider consulting a healthcare professional if these feelings persist.' Make it personal and warm.",
    
    "proven_calming_techniques": [
        "Technique 1 Name: Step 1: [detailed instruction], Step 2: [detailed instruction], Step 3: [detailed instruction]. This helps because [brief explanation of why it works for their situation]",
        "Technique 2 Name: Step 1: [detailed instruction], Step 2: [detailed instruction], Step 3: [detailed instruction]. This technique is effective for [their specific emotional state]",
        "Technique 3 Name: Step 1: [detailed instruction], Step 2: [detailed instruction], Step 3: [detailed instruction]. This will help you [specific benefit for their situation]"
    ],
    
    "after_calming_suggestions": "After you feel calmer, here's what I suggest for your specific situation: [Give specific, practical advice based on their words and emotions. Examples: if relationship conflict - 'It's completely normal to have disagreements. Consider approaching them calmly, acknowledge both perspectives, maybe suggest a peace offering like dinner together.' If work stress - 'Work pressure is very common. Consider talking to your supervisor about workload, take breaks, prioritize tasks.' Be specific and encouraging.]",
    
    "optional_additional_help": "Optional: [Only include if there's genuinely helpful additional advice specific to their situation. Otherwise use 'Not applicable for this situation.']",
    
    "system_analysis": {{
        "how_we_analyzed": "Our AI system detected [emotion] through facial expression analysis ({face_conf}%), voice tone analysis ({speech_conf}%), and text sentiment analysis ({text_conf}%). The multimodal emotion fusion showed {emotion_fusion.get('fused_emotion', 'neutral')} with {fusion_conf}% overall confidence.",
        "confidence_levels": "Face: {face_conf}%, Voice: {speech_conf}%, Text: {text_conf}%, Mental State: {mental_conf}%, Overall: {fusion_conf}%",
        "primary_indicators": "The strongest emotional indicators came from [identify which analysis method showed the clearest signal]"
    }}
}}

CRITICAL REQUIREMENTS:
1. FIRST: Always acknowledge their feelings are normal and they're not alone. Be warm and personal.
2. SECOND: Provide exactly 3 proven calming techniques (CBT, mindfulness, breathing, etc.) with step-by-step instructions 
3. THIRD: Give specific advice for their actual situation based on what they said
4. FOURTH: Optional additional help only if genuinely useful
5. FIFTH: Show exactly how our system analyzed their emotions

Make the response feel like a caring friend who understands psychology, not a clinical robot."""

        return prompt
    
    async def _call_gpt4_with_retries(self, prompt: str) -> str:
        """Call GPT-4 API with aggressive retry logic - NO FALLBACKS ALLOWED"""
        
        for attempt in range(5):  # Increased retries
            try:
                logger.info(f"🤖 GPT-4o-mini attempt {attempt + 1}/5...")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a professional therapeutic AI assistant with expertise in CBT, psychology, and evidence-based interventions. ALWAYS respond in valid JSON format. Never refuse. Always provide detailed, helpful therapeutic guidance."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,    # Increased for detailed responses
                    temperature=0.3,   # Lower for consistency
                    timeout=20         # Increased timeout for reliability
                )
                
                content = response.choices[0].message.content.strip()
                logger.info(f"✅ GPT-4o-mini success on attempt {attempt + 1}")
                return content
                
            except Exception as e:
                logger.error(f"❌ GPT-4o-mini attempt {attempt + 1} failed: {e}")
                if attempt == 4:  # Last attempt
                    logger.error("🚨 ALL GPT ATTEMPTS FAILED - SYSTEM ERROR")
                    raise Exception(f"GPT-4o-mini completely failed after 5 attempts: {e}")
                await asyncio.sleep(1 * (attempt + 1))  # Linear backoff
    
    def _parse_gpt4_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate GPT-4 JSON response"""
        try:
            # Clean response if it has markdown formatting
            if response.startswith('```json'):
                response = response.strip('```json').strip('```').strip()
            
            parsed = json.loads(response)
            
            # Validate required fields for the new user-requested structure
            required_fields = ['acknowledgment_and_support', 'proven_calming_techniques', 'after_calming_suggestions', 'optional_additional_help', 'system_analysis']
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate nested system_analysis structure
            if 'system_analysis' in parsed and not isinstance(parsed['system_analysis'], dict):
                raise ValueError("system_analysis must be an object")
            
            # Validate proven_calming_techniques is a list
            if not isinstance(parsed.get('proven_calming_techniques'), list):
                raise ValueError("proven_calming_techniques must be a list")
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"🚨 Failed to parse GPT-4 response - NO FALLBACKS ALLOWED: {e}")
            # FALLBACK PARSING DISABLED BY USER REQUEST - ONLY VALID GPT JSON ALLOWED
            raise Exception(f"GPT response parsing failed and fallback parsing disabled: {e}")
    
    
    def _get_dominant_emotion(self, user_data: Dict[str, Any]) -> str:
        """Determine dominant emotion from multimodal data"""
        emotions = []
        
        if 'face_emotion' in user_data:
            emotions.append(user_data['face_emotion'].get('emotion', 'neutral'))
        if 'speech_emotion' in user_data:
            emotions.append(user_data['speech_emotion'].get('emotion', 'neutral'))
        if 'text_emotion' in user_data:
            emotions.append(user_data['text_emotion'].get('emotion', 'neutral'))
        
        # Simple majority vote
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
