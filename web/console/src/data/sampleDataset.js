// Mock data for the Dataset Explorer (DS-01) until the API exposes the dataset.
// The sentences below are short, illustrative paragraphs written for the UI.
// They are NOT rows taken from WiLI-2018.

export const DATASET_STATS = {
  languages: 235,
  // Split sizes as agreed by the team (sum = 235,000 paragraphs).
  splits: [
    { id: 'train', label: 'Train', short: 'train', size: 117500 },
    { id: 'validation', label: 'Validation', short: 'val', size: 11750 },
    { id: 'test', label: 'Test', short: 'test', size: 105750 },
  ],
};

export const TOTAL_PARAGRAPHS = DATASET_STATS.splits.reduce((sum, s) => sum + s.size, 0);

// code = ISO 639-3 label, lang = BCP-47 tag for the browser (fonts, screen readers), dir = text direction.
const row = (id, code, name, lang, sentence, dir = 'ltr') => ({ id, code, name, lang, sentence, dir });

export const SAMPLES = {
  train: [
    row('tr-1', 'eng', 'English', 'en', 'The Mekong is a trans-boundary river in East Asia and Southeast Asia, flowing through China, Myanmar, Laos, Thailand, Cambodia and Vietnam before reaching the South China Sea.'),
    row('tr-2', 'vie', 'Vietnamese', 'vi', 'Hà Nội là thủ đô của Việt Nam và là thành phố đông dân thứ hai cả nước, nằm ở đồng bằng sông Hồng.'),
    row('tr-3', 'fra', 'French', 'fr', 'La tour Eiffel est une tour de fer située à Paris, sur le Champ-de-Mars, au bord de la Seine.'),
    row('tr-4', 'deu', 'German', 'de', 'Der Rhein ist einer der längsten und wichtigsten Flüsse Europas und entspringt in den Schweizer Alpen, bevor er in die Nordsee mündet.'),
    row('tr-5', 'jpn', 'Japanese', 'ja', '富士山は静岡県と山梨県にまたがる標高3776メートルの活火山で、日本で最も高い山として知られている。'),
    row('tr-6', 'spa', 'Spanish', 'es', 'El Amazonas es el río más caudaloso del mundo y atraviesa Brasil, Perú y Colombia antes de desembocar en el océano Atlántico.'),
    row('tr-7', 'ara', 'Arabic', 'ar', 'القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها، وتقع على ضفاف نهر النيل.', 'rtl'),
    row('tr-8', 'hin', 'Hindi', 'hi', 'गंगा भारत की सबसे पवित्र नदियों में से एक है, जो हिमालय से निकलकर बंगाल की खाड़ी में गिरती है।'),
    row('tr-9', 'rus', 'Russian', 'ru', 'Байкал — самое глубокое озеро на планете, расположенное на юге Восточной Сибири в России.'),
    row('tr-10', 'por', 'Portuguese', 'pt', 'Lisboa é a capital de Portugal e a cidade mais populosa do país, situada na margem do rio Tejo.'),
  ],
  validation: [
    row('va-1', 'ita', 'Italian', 'it', 'Il Colosseo è un grande anfiteatro situato nel centro della città di Roma e fu costruito nel primo secolo d.C.'),
    row('va-2', 'kor', 'Korean', 'ko', '서울은 대한민국의 수도이며 한강을 끼고 발달한 인구 천만 명에 가까운 대도시이다.'),
    row('va-3', 'zho', 'Chinese', 'zh', '长江是亚洲最长的河流，全长约六千三百公里，最终注入东海。'),
    row('va-4', 'tur', 'Turkish', 'tr', "İstanbul, Boğaziçi'nin iki yakasına yayılan ve Asya ile Avrupa kıtalarını birleştiren Türkiye'nin en kalabalık şehridir."),
    row('va-5', 'swe', 'Swedish', 'sv', 'Stockholm är Sveriges huvudstad och landets folkrikaste stad, belägen där Mälaren möter Östersjön.'),
    row('va-6', 'pol', 'Polish', 'pl', 'Wisła jest najdłuższą rzeką w Polsce i uchodzi do Morza Bałtyckiego niedaleko Gdańska.'),
    row('va-7', 'tha', 'Thai', 'th', 'กรุงเทพมหานครเป็นเมืองหลวงและเมืองที่มีประชากรมากที่สุดของประเทศไทย ตั้งอยู่ริมแม่น้ำเจ้าพระยา'),
    row('va-8', 'eng', 'English', 'en', 'Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into chemical energy stored in glucose.'),
  ],
  test: [
    row('te-1', 'ell', 'Greek', 'el', 'Η Αθήνα είναι η πρωτεύουσα της Ελλάδας και μία από τις αρχαιότερες πόλεις του κόσμου.'),
    row('te-2', 'nld', 'Dutch', 'nl', 'Amsterdam is de hoofdstad van Nederland en staat bekend om haar grachten en musea.'),
    row('te-3', 'vie', 'Vietnamese', 'vi', 'Phở là một món ăn truyền thống của Việt Nam, gồm bánh phở, nước dùng và thịt bò hoặc thịt gà.'),
    row('te-4', 'fra', 'French', 'fr', "Marie Curie est une physicienne et chimiste d'origine polonaise qui a reçu deux prix Nobel dans deux disciplines scientifiques différentes."),
    row('te-5', 'deu', 'German', 'de', 'Die Berliner Mauer teilte von 1961 bis 1989 die Stadt Berlin in einen östlichen und einen westlichen Teil.'),
    row('te-6', 'jpn', 'Japanese', 'ja', '京都は794年から1869年まで日本の首都であり、現在も多くの寺院や神社が残っている。'),
    row('te-7', 'spa', 'Spanish', 'es', 'Gabriel García Márquez fue un escritor colombiano, ganador del Premio Nobel de Literatura en 1982.'),
    row('te-8', 'hin', 'Hindi', 'hi', 'हिंदी भारत की राजभाषा है और इसे देवनागरी लिपि में लिखा जाता है।'),
    row('te-9', 'rus', 'Russian', 'ru', 'Москва является столицей России и крупнейшим городом страны, расположенным на реке Москве.'),
    row('te-10', 'ara', 'Arabic', 'ar', 'اللغة العربية من أكثر اللغات انتشارًا في العالم، ويتحدث بها مئات الملايين من الناس.', 'rtl'),
    row('te-11', 'eng', 'English', 'en', 'The Great Barrier Reef, off the coast of Queensland, Australia, is the largest coral reef system in the world.'),
  ],
};
