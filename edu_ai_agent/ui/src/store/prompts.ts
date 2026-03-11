import { atom } from 'jotai';
import { atomWithSessionStorage } from './utils';

export const promptAtom = atomWithSessionStorage<string>('prompts', [
  'Agent에게 인사하기',
])

export const isPromptModalAtom = atom<boolean>(false);