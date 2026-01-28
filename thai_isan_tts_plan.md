# Thai-Isan TTS Development Plan for Qwen3-TTS

## Executive Summary

This comprehensive plan outlines the development of Thai-Isan language support for the Qwen3-TTS system. Isan (Northeastern Thai) is a distinct language with 13-16 million native speakers, linguistically closer to Lao than Central Thai, with unique phonological characteristics that require specialized TTS handling.

## 1. Linguistic Research Summary ✓

### Thai Language Characteristics
- **Script**: Thai alphabet (44 consonants, 15 vowel symbols, 4 tone markers)
- **Tones**: 5 phonemic tones (mid, low, falling, high, rising)
- **Syllable Structure**: C(C)V(V)(C)+Tone
- **Consonants**: 20 initial consonants with 3 voice-onset times (voiced, tenuis, aspirated)
- **Vowels**: 27 vowel sounds including diphthongs and triphthongs

### Isan Language Characteristics
- **Classification**: Southwestern Tai language, Lao variety
- **Speakers**: 13-16 million native, 22 million total (L1+L2)
- **Mutual Intelligibility**: Only ~80% with Central Thai despite shared vocabulary
- **Status**: Officially classified as Thai dialect but linguistically distinct

### Key Phonological Differences from Central Thai

#### Consonants
- **Missing Sounds**: /tɕʰ/ and /ʃ/ → replaced with /s/
- **Rhotic /r/**: Rare → replaced with /h/ or /l/
- **Palatal Nasal**: Maintains /j/–/ɲ/ distinction (merged to /j/ in Thai)
- **Clusters**: Rare compared to Thai
- **Labial Approximant**: /ʋ/ allophone for /w/ (not in Thai)

#### Vowels
- **Centralization**: /ɯ/ → /ɨ/, /ɤ/ → /ə/
- **Nasal Quality**: Perceived as nasal by Thai speakers
- **Length**: Extended diphthongs (e.g., /tuːa/ vs Thai /tua/)

#### Tonal System
- **Number**: 5-6 tones depending on dialect (Vientiane, Luang Phrabang, etc.)
- **Categories**: High, Middle, Low class consonants with different tonal patterns
- **Syllable Types**: Live vs. dead syllables affect tone realization
- **Tone Marks**: Mai ek, Mai tho influence tone patterns differently than Thai

## 2. Qwen3-TTS Architecture Analysis ✓

### Current Capabilities
- **Languages**: 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- **Architecture**: Discrete multi-codebook LM with Qwen3-TTS-Tokenizer-12Hz
- **Features**: Voice cloning, voice design, instruction control, streaming generation
- **Tokenizers**: 12Hz and 25Hz variants for speech tokenization

### Extension Requirements for Thai-Isan
- **Vocabulary**: Expand tokenizer vocabulary for Thai/Isan phonemes and graphemes
- **Phoneme Mapping**: Create Thai-Isan phoneme conversion rules
- **Tone Modeling**: Extend acoustic model for 5-6 tone system
- **Language ID**: Add Thai and Isan language identifiers
- **Cultural Context**: Handle cultural expressions and regional vocabulary

## 3. Thai-Isan Speech Dataset Collection and Preprocessing Plan

### Dataset Requirements
- **Minimum Duration**: 100+ hours per language (Thai, Isan)
- **Speakers**: Minimum 50 speakers per language (balanced gender, age, region)
- **Quality**: 48kHz sampling rate, studio-quality recordings
- **Annotation**: Phoneme-level, word-level, and sentence-level alignment

### Data Sources
1. **Public Corpora**:
   - NST Thai Speech Dataset
   - SEAME Mandarin-English Code-switching (adaptable)
   - Common Voice Thai (Mozilla)

2. **Academic Partnerships**:
   - Mahidol University (Thailand)
   - Chulalongkorn University
   - Khon Kaen University (Isan region)

3. **Commercial Sources**:
   - Local radio stations (Isan region)
   - Podcast recordings
   - Audiobook narrations

4. **Field Recording**:
   - Native Isan speakers from different provinces
   - Traditional story tellers and local performers
   - Different age groups and education levels

### Preprocessing Pipeline
```python
# Proposed preprocessing steps
1. Audio Quality Assessment
   - SNR calculation and filtering (>20dB)
   - Clipping detection and removal
   - Background noise reduction

2. Text Normalization
   - Thai/Isan script standardization
   - Number expansion
   - Abbreviation handling

3. Phoneme Alignment
   - Forced alignment using Montreal Forced Aligner
   - Manual correction for tone boundaries
   - Cross-validation by native speakers

4. Data Augmentation
   - Speed variation (0.9x-1.1x)
   - Pitch shifting (±2 semitones)
   - Reverberation and background noise
```

### Regional Isan Variants to Include
- **Northern Isan** (Loei, Nong Khai): Influenced by Luang Prabang Lao
- **Central Isan** (Khon Kaen, Kalasin): Standard Isan
- **Southern Isan** (Ubon Ratchathani, Sisaket): Influenced by Pakse Lao
- **Western Isan** (Nakhon Phanom, Sakon Nakhon): Close to Lao standard

## 4. Language Support Extension Strategy

### Phase 1: Thai Language Integration
```python
# Configuration extension
class ThaiIsanConfig(Qwen3TTSConfig):
    languages = ["th", "tts"]  # th=Thai, tts=Isan
    phoneme_inventory = {
        "th": get_thai_phonemes(),
        "tts": get_isan_phonemes()
    }
    tone_systems = {
        "th": 5,  # 5 tones
        "tts": 6  # 6 tones for Isan
    }
```

### Phase 2: Phoneme Mapping System
```python
# Thai-Isan phoneme conversion
THAI_TO_ISAN_MAPPINGS = {
    # Consonant substitutions
    "tɕʰ": "s",      # Thai ช → Isan s
    "r": ["h", "l"],  # Thai r → Isan h/l
    "j": {"context": "if_original_ɲ", "output": "ɲ"},
    
    # Vowel substitutions
    "ɯ": "ɨ",        # Centralization
    "ɤ": "ə",        # Centralization
    
    # Tone mappings
    "thai_tone_1": "isan_tone_low",
    "thai_tone_2": "isan_tone_falling",
}
```

### Phase 3: Tokenizer Extension
- Extend vocabulary to 160k+ tokens for Thai/Isan support
- Add subword units for Thai script complexity
- Include tone markers as separate tokens
- Support Lao script variants for Isan

## 5. Phoneme Mapping and Pronunciation Rules

### Thai Phoneme Inventory (44 consonants, 21 sounds)
```
Labial: /m/, /p/, /pʰ/, /b/, /f/
Dental: /n/, /t/, /tʰ/, /d/, /s/
Postalveolar: /tɕ/, /tɕʰ/
Velar: /ŋ/, /k/, /kʰ/
Glottal: /ʔ/, /h/
Approximants: /l/, /j/, /w/
```

### Isan Phoneme Inventory (Key Differences)
```
# Missing sounds
/tɕʰ/ → /s/ (Thai ช sounds like Isan s)
/r/ → /h/ or /l/

# Additional distinctions
/ɲ/ vs /j/ (maintained in Isan, merged in Thai)
/ʋ/ allophone for /w/

# Vowel differences
Close back unrounded /ɯ/ → centralized /ɨ/
Close-mid back unrounded /ɤ/ → mid central /ə/
```

### Tone System Mapping
```python
# Thai tone categories
HIGH_CLASS = ["ข", "ฉ", "ฐ", "ถ", "ผ", "ฝ", "ศ", "ส", "ห"]
MID_CLASS = ["ก", "จ", "ฎ", "ฏ", "ด", "ต", "บ", "ป", "อ"]
LOW_CLASS = ["ค", "ฅ", "ฆ", "ง", "ช", "ซ", "ฌ", "ญ", "ฑ", "ฒ", "ณ", "ท", "ธ", "น", "พ", "ฟ", "ภ", "ม", "ย", "ร", "ล", "ว", "ฬ", "ส"]

# Tone realization rules
TONE_MAPPING = {
    "thai": {
        "high_long": "rising",
        "high_short": "low",
        "mid_live": "mid",
        "mid_dead": "low",
        "low_live": "mid",
        "low_dead": "falling"
    },
    "isan": {
        "high_long": "high_falling",
        "high_short": "mid",
        "mid_live": "mid",
        "mid_dead": "low",
        "low_live": "rising",
        "low_dead": "high_falling"
    }
}
```

## 6. Fine-tuning and Training Pipeline

### Training Strategy
1. **Phase 1**: Pre-train on combined Thai+Isan dataset
2. **Phase 2**: Fine-tune separate heads for each language
3. **Phase 3**: Joint optimization with language-adversarial training

### Model Architecture Modifications
```python
class ThaiIsanTTSModel(Qwen3TTSModel):
    def __init__(self, config):
        super().__init__(config)
        # Add language-specific layers
        self.thai_tone_predictor = TonePredictor(num_tones=5)
        self.isan_tone_predictor = TonePredictor(num_tones=6)
        self.phoneme_adapter = PhonemeAdapter(
            thai_phonemes=44, 
            isan_phonemes=42
        )
    
    def forward(self, text, language="th"):
        if language == "tts":
            # Apply Isan-specific processing
            phonemes = self.thai_to_isan_converter(text)
            tones = self.isan_tone_predictor(phonemes)
        else:
            # Standard Thai processing
            phonemes = self.thai_processor(text)
            tones = self.thai_tone_predictor(phonemes)
        
        return self.generate_speech(phonemes, tones)
```

### Training Configuration
```yaml
# Training parameters
batch_size: 32
learning_rate: 2e-5
num_epochs: 100
warmup_steps: 1000
gradient_accumulation_steps: 4

# Language-specific weights
loss_weights:
  thai: 1.0
  isan: 1.2  # Higher weight for minority language

# Data mixing strategy
data_ratio:
  thai: 0.6
  isan: 0.4
```

## 7. Voice Samples and Speaker Profiles

### Speaker Categories
1. **Regional Isan Speakers**:
   - Northern Isan (Loei, Nong Khai): 10 speakers
   - Central Isan (Khon Kaen): 10 speakers
   - Southern Isan (Ubon): 10 speakers
   - Western Isan (Nakhon Phanom): 10 speakers

2. **Thai Speakers** (for comparison):
   - Central Thai: 15 speakers
   - Southern Thai: 10 speakers
   - Northern Thai: 10 speakers

3. **Demographic Balance**:
   - Gender: 50% male, 50% female
   - Age: 20-30 (30%), 30-50 (40%), 50+ (30%)
   - Education: Various levels
   - Occupation: Various backgrounds

### Voice Profile Features
```python
VOICE_PROFILES = {
    "isan_northern_male_30": {
        "region": "Loei",
        "age": 30,
        "gender": "male",
        "education": "bachelor",
        "occupation": "teacher",
        "features": ["clear_articulation", "moderate_tone_range"],
        "sample_rate": 48000,
        "duration_hours": 5
    },
    "isan_central_female_45": {
        "region": "Khon Kaen",
        "age": 45,
        "gender": "female",
        "education": "high_school",
        "occupation": "merchant",
        "features": ["natural_prosody", "local_expressions"],
        "sample_rate": 48000,
        "duration_hours": 4
    }
}
```

## 8. Evaluation Metrics for Thai-Isan TTS

### Objective Metrics
1. **Speech Quality**:
   - Mel-Cepstral Distortion (MCD)
   - Fundamental Frequency (F0) correlation
   - Voice conversion similarity

2. **Language-Specific Metrics**:
   - Tone accuracy (95%+ target)
   - Phoneme error rate (PER < 5%)
   - Consonant cluster handling
   - Vowel length accuracy

### Subjective Metrics
1. **Mean Opinion Score (MOS)**:
   - Naturalness: 1-5 scale (target > 4.0)
   - Intelligibility: 1-5 scale (target > 4.2)
   - Authenticity: 1-5 scale (target > 3.8)

2. **Language-Specific Evaluation**:
   - Native speaker preference
   - Regional accent recognition
   - Cultural appropriateness
   - Emotional expressiveness

### Thai-Isan Specific Tests
```python
# Test sentences covering key features
TEST_SENTENCES = {
    "tone_distinction": [
        "กา", "ก่า", "ก้า", "ก๊า", "ก๋า",  # Thai tones
        "ກາ", "ກ້າ", "ກ๊າ", "ກ໋າ", "ກ່າ"   # Isan tones
    ],
    "consonant_substitutions": [
        "ช้าง" → "ส้าง",  # /tɕʰ/ → /s/
        "ร้อน" → "ฮ้อน",  # /r/ → /h/
        "ยาก" → "หญาก"   # /j/ → /ɲ/
    ],
    "vowel_centralization": [
        "ขึ้น" → "ขึ้น",  # /ɯ/ maintained
        "เดิน" → "เดิน"   # /ɤ/ centralized
    ],
    "regional_variations": [
        "เฮา" (Isan) vs "เรา" (Thai),  # "we/us"
        "เจ้า" vs "คุณ",  # "you"
        "บ่" vs "ไม่"     # "not"
    ]
}
```

## 9. Web Demo Interface Design

### Features
1. **Language Selection**: Thai, Isan, Auto-detect
2. **Regional Variants**: Northern, Central, Southern, Western Isan
3. **Voice Options**: Multiple speakers per region
4. **Text Input**: Unicode support for Thai/Lao scripts
5. **Prosody Control**: Speed, pitch, emotion
6. **Comparison Mode**: Thai vs Isan pronunciation

### Technical Implementation
```javascript
// Frontend components
const ThaiIsanTTSInterface = {
  languages: ['th', 'tts'],
  regions: {
    'tts': ['northern', 'central', 'southern', 'western']
  },
  
  // Script conversion utilities
  convertThaiToIsan: function(text) {
    // Apply phoneme and tone conversions
    return convertedText;
  },
  
  // Voice synthesis
  synthesizeSpeech: async function(text, language, region, voice) {
    const response = await fetch('/api/tts', {
      method: 'POST',
      body: JSON.stringify({ text, language, region, voice })
    });
    return response.blob();
  }
};
```

## 10. Documentation Structure

### Technical Documentation
1. **API Reference**: Complete method documentation
2. **Training Guide**: Step-by-step training instructions
3. **Deployment Manual**: Production deployment guidelines
4. **Performance Benchmarks**: Speed and quality metrics

### User Documentation
1. **Quick Start Guide**: 5-minute setup
2. **Language Support**: Thai vs Isan capabilities
3. **Voice Samples**: Audio examples
4. **Troubleshooting**: Common issues and solutions

### Academic Documentation
1. **Linguistic Analysis**: Phonological comparison
2. **Dataset Description**: Collection methodology
3. **Evaluation Results**: Comprehensive testing
4. **Future Work**: Research directions

## 11. Implementation Timeline

### Phase 1: Foundation (Weeks 1-4)
- [ ] Dataset collection and preprocessing
- [ ] Phoneme mapping system development
- [ ] Basic Thai language support

### Phase 2: Core Development (Weeks 5-8)
- [ ] Isan language integration
- [ ] Tone system implementation
- [ ] Voice sample collection

### Phase 3: Training and Optimization (Weeks 9-12)
- [ ] Model training and fine-tuning
- [ ] Performance optimization
- [ ] Quality evaluation

### Phase 4: Deployment and Testing (Weeks 13-16)
- [ ] Web demo development
- [ ] User testing and feedback
- [ ] Documentation completion

## 12. Success Metrics

### Technical Targets
- **Tone Accuracy**: >95% correct tone realization
- **Phoneme Accuracy**: <5% phoneme error rate
- **Naturalness**: MOS >4.0 for both languages
- **Latency**: <200ms for real-time synthesis
- **Coverage**: Support for all regional Isan variants

### User Experience Targets
- **Ease of Use**: <5 minutes setup time
- **Voice Quality**: >80% user preference over existing solutions
- **Cultural Authenticity**: Recognized by native speakers
- **Regional Accuracy**: Correct regional accent identification

### Business Impact
- **Market Coverage**: 22 million potential users
- **Competitive Advantage**: First comprehensive Thai-Isan TTS
- **Academic Recognition**: Contributions to Southeast Asian linguistics
- **Cultural Preservation**: Documentation of endangered language features

## 13. Risk Assessment and Mitigation

### Technical Risks
1. **Insufficient Training Data**: Partner with universities, use data augmentation
2. **Tone Modeling Complexity**: Gradual complexity increase, expert consultation
3. **Regional Variation Handling**: Multi-speaker training, regional fine-tuning

### Linguistic Risks
1. **Language Shift**: Focus on younger speakers, modern vocabulary
2. **Script Standardization**: Support both Thai and Lao scripts
3. **Cultural Sensitivity**: Collaborate with local communities

### Market Risks
1. **Limited Commercial Interest**: Focus on educational/government applications
2. **Competition from Generic Solutions**: Emphasize cultural authenticity
3. **Technology Adoption**: Provide comprehensive training and support

## 14. Future Enhancements

### Short-term (6 months)
- Code-switching between Thai and Isan
- Emotional expression control
- Real-time voice conversion

### Medium-term (1 year)
- Lao language support
- Northern/Southern Thai dialects
- Mobile application

### Long-term (2+ years)
- Other Tai-Kadai languages
- Cross-lingual voice transfer
- Emotion-preserving translation

This comprehensive plan provides a roadmap for developing Thai-Isan TTS support in Qwen3-TTS, addressing both technical challenges and cultural considerations while maintaining the high-quality standards of the existing system.