import re
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab


PAGES_DIR = Path(__file__).parent.parent / 'task1' / 'pages'
OUTPUT_DIR = Path(__file__).parent / 'output'
CONJUNCTIONS = {
    'и', 'а', 'но', 'или', 'однако', 'зато', 'если', 'коли', 'когда',
    'лишь', 'только', 'что', 'чтобы', 'как', 'ежели', 'нежели', 'хотя',
    'хоть', 'пусть', 'раз', 'кабы', 'коль', 'дабы', 'ибо', 'ведь',
    'же', 'ли', 'да', 'неужели', 'неужто', 'также', 'тоже', 'причем',
    'причём', 'словно', 'будто', 'что-то', 'кое-что'
}
PREPOSITIONS = {
    'в', 'на', 'с', 'к', 'по', 'из', 'за', 'от', 'без', 'через',
    'для', 'при', 'о', 'об', 'у', 'под', 'над', 'между', 'перед',
    'около', 'вокруг', 'возле', 'после', 'вследствие', 'благодаря',
    'вопреки', 'наперекор', 'исключая', 'включая', 'ради', 'кроме',
    'подобно', 'согласно', 'посреди', 'среди', 'внутри', 'вне',
    'позади', 'напротив', 'против', 'из-за', 'из-под', 'мимо',
    'сквозь', 'близ', 'у', 'воз', 'к', 'поперек', 'напоперек'
}
STOP_WORDS = CONJUNCTIONS | PREPOSITIONS


segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def extract_text_from_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='windows-1251', errors='replace') as f:
            html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    text_parts = []
    for z_tag in soup.find_all('z'):
        for o_tag in z_tag.find_all('o'):
            parent = o_tag.parent
            o_tag.decompose()
            text_parts.append(parent.get_text(separator=' ', strip=True))
    
    return ' '.join(text_parts)


def is_valid_token(token):
    token_lower = token.lower()
    
    if re.match(r'^\d+$', token_lower):
        return False
    
    if re.search(r'\d.*[а-яА-ЯёЁ]|[а-яА-ЯёЁ].*\d', token_lower):
        return False
    
    if re.match(r'^[^а-яА-ЯёЁ]+$', token_lower):
        return False
    
    if not re.search(r'[а-яА-ЯёЁ]', token_lower):
        return False
    
    if len(re.findall(r'[а-яА-ЯёЁ]', token_lower)) <= 1:
        return False
    
    if re.search(r'[<>&⟨⟩{}\[\]\\/|]', token_lower):
        return False
    
    return True


def split_hyphenated(token):
    parts = re.split(r'[-—]', token)
    return [p.strip() for p in parts if p.strip()]


def process_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    tokens_set = set()
    lemma_to_tokens = defaultdict(set)
    inverted_index = defaultdict(list)

    for token in doc.tokens:
        original = token.text.strip('.,;:!?()[]""\'"…«»—')
        if not original:
            continue
        
        if not is_valid_token(original):
            continue
        
        original_lower = original.lower()
        
        if original_lower in STOP_WORDS:
            continue
        
        token.lemmatize(morph_vocab)
        lemma = token.lemma.lower() if token.lemma else original_lower
        
        parts = split_hyphenated(original_lower)

        for part in parts:
            if not part:
                continue
            
            if not is_valid_token(part):
                continue
            
            if part in STOP_WORDS:
                continue
            
            tokens_set.add(part)
            
            part_doc = Doc(part)
            part_doc.segment(segmenter)
            part_doc.tag_morph(morph_tagger)
            
            if part_doc.tokens:
                part_doc.tokens[0].lemmatize(morph_vocab)
                part_lemma = part_doc.tokens[0].lemma.lower() if part_doc.tokens[0].lemma else part
            else:
                part_lemma = part
            
            lemma_to_tokens[part_lemma].add(part)
            inverted_index[part_lemma].append(token.start)
    
    return tokens_set, lemma_to_tokens, inverted_index


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    html_files = sorted(PAGES_DIR.glob('*.html'), key=lambda x: int(x.stem))
    
    print(f'Processing {len(html_files)} HTML files...')
    
    for filepath in html_files:
        file_num = filepath.stem
        print(f'Processing {filepath.name}...', end='\r')
        
        text = extract_text_from_html(filepath)
        
        if text:
            tokens, lemma_to_tokens, _ = process_text(text)
            
            file_output_dir = OUTPUT_DIR / file_num
            file_output_dir.mkdir(exist_ok=True)
            
            tokens_path = file_output_dir / 'tokens.txt'
            with open(tokens_path, 'w', encoding='utf-8') as f:
                for token in sorted(tokens):
                    f.write(f'{token}\n')
            
            lemmas_path = file_output_dir / 'lemmas.txt'
            with open(lemmas_path, 'w', encoding='utf-8') as f:
                for lemma in sorted(lemma_to_tokens.keys()):
                    tokens_str = ' '.join(sorted(lemma_to_tokens[lemma]))
                    f.write(f'{lemma} {tokens_str}\n')


if __name__ == '__main__':
    main()
