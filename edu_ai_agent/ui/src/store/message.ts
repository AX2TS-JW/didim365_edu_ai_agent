import { atom } from 'jotai';
import { atomWithSessionStorage } from './utils';

export const messageAtom = atomWithSessionStorage<string>('message', '')

export const threadById = atom<string>('');