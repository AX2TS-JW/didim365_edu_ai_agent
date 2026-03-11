import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import ko from './ko/ko.json';
import en from './en/en.json';

const dlng = !window.navigator.language.match(/en|ko/i) ? 'en' : window.navigator.language;

const resources = Object.freeze({
    en: {translation: en},
    ko: {translation: ko},
});

// when it is deployed, its working great.
i18n
  .use(initReactI18next)
  .init({
    lng: dlng,
    debug: false,
    fallbackLng: ['en', 'ko'],
    interpolation: {
      escapeValue: false
    },
    // load: 'languageOnly',
    supportedLngs: ['en', 'ko'],
    resources,
    /* order and from where user language should be detected */
    detection: {
      order: ['navigator'],
      // caches: ['localStorage']
    }
});

export default i18n;