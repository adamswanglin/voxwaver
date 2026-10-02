import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import ar from './locales/ar'
import de from './locales/de'
import en from './locales/en'
import es from './locales/es'
import fr from './locales/fr'
import it from './locales/it'
import ja from './locales/ja'
import ko from './locales/ko'
import nl from './locales/nl'
import pl from './locales/pl'
import pt from './locales/pt'
import ru from './locales/ru'
import zh from './locales/zh'

/** Boot cache only — the Rust settings file is the source of truth; the
 * mirror just lets init pick the right language synchronously so non-zh
 * users don't get a Chinese flash before settings load. */
const LANG_KEY = 'vw.uiLang'
const bootLang = localStorage.getItem(LANG_KEY) || 'zh'

void i18n.use(initReactI18next).init({
  resources: {
    zh: { translation: zh },
    en: { translation: en },
    ja: { translation: ja },
    de: { translation: de },
    fr: { translation: fr },
    es: { translation: es },
    ko: { translation: ko },
    ar: { translation: ar },
    ru: { translation: ru },
    nl: { translation: nl },
    it: { translation: it },
    pl: { translation: pl },
    pt: { translation: pt },
  },
  lng: bootLang,
  fallbackLng: 'zh',
  interpolation: { escapeValue: false }, // React already escapes
  returnEmptyString: false,
})

/** Switch UI language and sync the boot cache + document attributes. */
export function applyLanguage(l: string) {
  void i18n.changeLanguage(l)
  localStorage.setItem(LANG_KEY, l)
  document.documentElement.lang = l
  document.documentElement.dir = l === 'ar' ? 'rtl' : 'ltr'
}

export default i18n
