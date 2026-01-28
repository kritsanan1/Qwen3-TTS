"""
Thai-Isan Language Support for Qwen3-TTS
Core phoneme mapping and pronunciation rules
"""

# Thai consonant inventory (44 graphemes, 21 phonemes)
THAI_CONSONANTS = {
    'labial': {
        'nasal': ['ม', 'หม'],
        'plosive_voiceless': ['ป', 'ผ', 'พ', 'ภ'],
        'plosive_voiced': ['บ'],
        'fricative': ['ฝ', 'ฟ']
    },
    'dental': {
        'nasal': ['น', 'ณ', 'หน', 'หณ'],
        'plosive_voiceless': ['ต', 'ฏ', 'ถ', 'ฐ', 'ฑ', 'ฒ', 'ท', 'ธ'],
        'plosive_voiced': ['ด', 'ฎ'],
        'fricative': ['ซ', 'ศ', 'ษ', 'ส']
    },
    'alveolar': {
        'approximant': ['ล', 'ฬ', 'หล', 'หฬ'],
        'trill': ['ร', 'หร']
    },
    'postalveolar': {
        'affricate': ['จ', 'ฉ', 'ช', 'ฌ']
    },
    'velar': {
        'nasal': ['ง', 'หง'],
        'plosive_voiceless': ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ'],
        'fricative': ['ข', 'ฃ', 'ค', 'ฅ', 'ฆ']  # [x] allophone
    },
    'glottal': {
        'plosive': ['อ'],
        'fricative': ['ห', 'ฮ']
    },
    'palatal': {
        'approximant': ['ย', 'หย']
    }
}

# Isan consonant differences from Thai
ISAN_CONSONANT_DIFFERENCES = {
    # Thai sounds missing in Isan
    'tɕʰ': 's',      # Thai ช → Isan s (e.g., ช้าง → ส้าง)
    'ʃ': 's',        # Allophone of /tɕʰ/ → /s/
    
    # Rhotic differences
    'r': ['h', 'l'],  # Thai r → Isan h/l (e.g., ร้อน → ฮ้อน)
    
    # Palatal nasal distinction maintained
    'j_context_ɲ': 'ɲ',  # Maintain /ɲ/ when etymologically present
    
    # Labial approximant variation
    'w': ['w', 'ʋ'],  # Allophonic variation in Isan
}

# Thai vowel inventory (15 symbols, 27 sounds)
THAI_VOWELS = {
    'short': {
        'front': ['ิ', 'ึ'],
        'central': ['ึ'],
        'back': ['ุ']
    },
    'long': {
        'front': ['ี', 'ื'],
        'central': ['ือ', 'เ', 'แ'],
        'back': ['ู', 'โ', 'ะ', 'า']
    },
    'diphthongs': ['ำ', 'ไ', 'ใ', 'เา', 'าว']
}

# Isan vowel differences
ISAN_VOWEL_DIFFERENCES = {
    # Centralization of back vowels
    'ɯ': 'ɨ',    # Close back unrounded → close central unrounded
    'ɤ': 'ə',    # Close-mid back unrounded → mid central
    
    # Nasal quality (perceptual)
    'nasalization': 'add_nasal_quality',
    
    # Length differences in diphthongs
    'ua': 'uːa',  # Lengthen first element
    'ia': 'iːa',
    'ɯa': 'ɨːa'
}

# Tone classification system
THAI_TONE_CLASSES = {
    'high_class': ['ข', 'ฃ', 'ฉ', 'ฐ', 'ถ', 'ผ', 'ฝ', 'ศ', 'ษ', 'ส', 'ห'],
    'mid_class': ['ก', 'จ', 'ฎ', 'ฏ', 'ด', 'ต', 'บ', 'ป', 'อ'],
    'low_class': ['ค', 'ฅ', 'ฆ', 'ง', 'ช', 'ซ', 'ฌ', 'ญ', 'ฑ', 'ฒ', 'ณ', 'ท', 'ธ', 'น', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ฬ', 'ฮ']
}

THAI_TONE_MARKERS = {
    '่': 'mai_ek',      # Low tone mark
    '้': 'mai_tho',      # Falling tone mark
    '๊': 'mai_tri',      # High tone mark
    '๋': 'mai_chattawa'  # Rising tone mark
}

# Tone realization rules for Thai
THAI_TONE_RULES = {
    'high_class': {
        'live_syllable': 'rising',
        'dead_short': 'low',
        'dead_long': 'low',
        'with_mai_ek': 'low',
        'with_mai_tho': 'falling'
    },
    'mid_class': {
        'live_syllable': 'mid',
        'dead_short': 'low',
        'dead_long': 'low',
        'with_mai_ek': 'low',
        'with_mai_tho': 'falling'
    },
    'low_class': {
        'live_syllable': 'mid',
        'dead_short': 'high',
        'dead_long': 'falling',
        'with_mai_ek': 'falling',
        'with_mai_tho': 'high'
    }
}

# Isan tone system (6 tones in some dialects)
ISAN_TONE_RULES = {
    'vientiane_dialect': {
        'high_class': {
            'live_syllable': 'high_falling',
            'dead_short': 'mid',
            'dead_long': 'mid',
            'with_mai_ek': 'mid',
            'with_mai_tho': 'high_falling'
        },
        'mid_class': {
            'live_syllable': 'mid',
            'dead_short': 'low',
            'dead_long': 'low',
            'with_mai_ek': 'low',
            'with_mai_tho': 'high_falling'
        },
        'low_class': {
            'live_syllable': 'rising',
            'dead_short': 'high_falling',
            'dead_long': 'high_falling',
            'with_mai_ek': 'high_falling',
            'with_mai_tho': 'rising'
        }
    },
    'luang_prabang_dialect': {
        # Different tone realizations
        'tone_count': 5,
        'patterns': 'slightly_different'
    }
}

# Common Isan vocabulary differences from Thai
ISAN_VOCABULARY_DIFFERENCES = {
    'pronouns': {
        'เรา': 'เฮา',      # we/us
        'คุณ': 'เจ้า',     # you (formal)
        'ฉัน': 'ข้า',      # I (female)
        'ผม': 'ข้า',       # I (male)
    },
    'negation': {
        'ไม่': 'บ่',       # not
        'ไม่ได้': 'บ่ได้',   # cannot
    },
    'common_words': {
        'กิน': 'แซบ',      # eat
        'สวย': 'งาม',      # beautiful
        'ไป': 'ไป๋',       # go
        'มา': 'มา๋',       # come
        'น้ำ': 'น้ำ',      # water (same)
        'ข้าว': 'ข้าว',    # rice (same)
    },
    'final_particles': {
        'ครับ': 'เด้อ',    # male polite
        'ค่ะ': 'เด้อ',     # female polite
        'จ้า': 'จ้า',      # friendly
    }
}

# Phoneme conversion from Thai to Isan
def thai_to_isan_phoneme_conversion(thai_phoneme, context=None):
    """
    Convert Thai phonemes to Isan equivalents
    
    Args:
        thai_phoneme: Thai phoneme string
        context: Optional context information
    
    Returns:
        Isan phoneme equivalent
    """
    # Direct substitutions
    if thai_phoneme in ISAN_CONSONANT_DIFFERENCES:
        replacement = ISAN_CONSONANT_DIFFERENCES[thai_phoneme]
        if isinstance(replacement, list):
            # Context-dependent selection
            if context and context.get('position') == 'initial':
                return replacement[0]  # Prefer first option
            else:
                return replacement[1] if len(replacement) > 1 else replacement[0]
        else:
            return replacement
    
    # Context-dependent substitutions
    if thai_phoneme == 'j' and context:
        if context.get('etymological') == 'ɲ':
            return 'ɲ'
    
    # Vowel centralization
    if thai_phoneme in ISAN_VOWEL_DIFFERENCES:
        return ISAN_VOWEL_DIFFERENCES[thai_phoneme]
    
    # Default: return original
    return thai_phoneme

# Tone conversion from Thai to Isan
def thai_to_isan_tone_conversion(thai_tone, consonant_class, syllable_type, dialect='vientiane'):
    """
    Convert Thai tone to Isan equivalent
    
    Args:
        thai_tone: Thai tone name
        consonant_class: High, Mid, or Low class
        syllable_type: Live, dead_short, dead_long
        dialect: Isan dialect variant
    
    Returns:
        Isan tone equivalent
    """
    if dialect not in ISAN_TONE_RULES:
        dialect = 'vientiane_dialect'
    
    rules = ISAN_TONE_RULES[dialect][consonant_class]
    
    # Map syllable type to rule key
    if syllable_type == 'live':
        rule_key = 'live_syllable'
    elif syllable_type == 'dead_short':
        rule_key = 'dead_short'
    elif syllable_type == 'dead_long':
        rule_key = 'dead_long'
    else:
        rule_key = 'live_syllable'  # Default
    
    return rules.get(rule_key, thai_tone)  # Default to original if not found

# Word-level conversion from Thai to Isan
def thai_to_isan_word_conversion(thai_word):
    """
    Convert Thai words to Isan equivalents
    
    Args:
        thai_word: Thai word in Thai script
    
    Returns:
        Isan word equivalent
    """
    # Check vocabulary substitutions
    for category, mappings in ISAN_VOCABULARY_DIFFERENCES.items():
        if thai_word in mappings:
            return mappings[thai_word]
    
    # If no vocabulary substitution, apply phonological rules
    # This would require syllable parsing and phoneme conversion
    return thai_word  # Placeholder

# Syllable parsing for Thai/Isan
def parse_syllable(word):
    """
    Parse Thai/Isan syllable structure
    
    Returns:
        Dictionary with syllable components
    """
    # This is a simplified parser
    # Real implementation would require comprehensive Thai syllable grammar
    
    syllable = {
        'initial_consonant': None,
        'vowel': None,
        'final_consonant': None,
        'tone_mark': None,
        'tone_class': None
    }
    
    # Parse tone marks
    for mark, name in THAI_TONE_MARKERS.items():
        if mark in word:
            syllable['tone_mark'] = name
            break
    
    # Parse initial consonant
    for consonant_class, consonants in THAI_TONE_CLASSES.items():
        for consonant in consonants:
            if word.startswith(consonant):
                syllable['initial_consonant'] = consonant
                syllable['tone_class'] = consonant_class
                break
    
    return syllable

# Example usage and testing
if __name__ == "__main__":
    # Test phoneme conversions
    print("Converting /tɕʰ/ to Isan:", thai_to_isan_phoneme_conversion('tɕʰ'))
    print("Converting /r/ to Isan:", thai_to_isan_phoneme_conversion('r'))
    print("Converting /ɯ/ to Isan:", thai_to_isan_phoneme_conversion('ɯ'))
    
    # Test tone conversions
    print("Converting Thai tone to Isan:", 
          thai_to_isan_tone_conversion('mid', 'high_class', 'live'))
    
    # Test word conversions
    print("Converting 'เรา' to Isan:", thai_to_isan_word_conversion('เรา'))
    print("Converting 'ไม่' to Isan:", thai_to_isan_word_conversion('ไม่'))
    
    # Test syllable parsing
    test_word = "ช้าง"
    parsed = parse_syllable(test_word)
    print(f"Parsed syllable for '{test_word}':", parsed)