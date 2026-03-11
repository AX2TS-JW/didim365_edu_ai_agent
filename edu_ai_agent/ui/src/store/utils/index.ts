import { atomWithStorage, createJSONStorage } from 'jotai/utils';

export const atomWithSessionStorage = <T>(key, initialValue) => {
  return atomWithStorage(
    key,
    initialValue,
    createJSONStorage(() => sessionStorage)
  )
}