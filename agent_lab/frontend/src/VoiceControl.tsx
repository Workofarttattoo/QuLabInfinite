import React, { useState, useEffect } from 'react';
import axios from 'axios';

const VoiceControl = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [recognition, setRecognition] = useState(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = true;
      recognitionInstance.interimResults = true;
      setRecognition(recognitionInstance);
    } else {
      console.error("Speech recognition not supported in this browser.");
    }
  }, []);

  const handleListen = () => {
    if (isListening) {
      recognition.stop();
      setIsListening(false);
      if (transcript) {
        // Send the final transcript to the backend
        axios.post('http://localhost:8000/broadcast/hearing', { text: transcript })
          .then(() => console.log("Broadcasted:", transcript))
          .catch(err => console.error("Error broadcasting:", err));
      }
    } else {
      setIsListening(true);
      recognition.start();
    }
  };

  useEffect(() => {
    if (!recognition) return;

    recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';
      for (let i = 0; i < event.results.length; i++) {
        const transcriptPart = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcriptPart;
        } else {
          interimTranscript += transcriptPart;
        }
      }
      setTranscript(finalTranscript || interimTranscript);
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
    };

  }, [recognition]);

  return (
    <div style={{ padding: '10px', borderTop: '1px solid #ddd' }}>
      <h4>Voice Control</h4>
      <button onClick={handleListen}>
        {isListening ? 'Stop Listening' : 'Start Listening'}
      </button>
      <p><strong>Transcript:</strong> {transcript}</p>
    </div>
  );
};

export default VoiceControl;
