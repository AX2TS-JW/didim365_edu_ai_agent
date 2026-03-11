import { IMessage } from '../types/chatVM';
import { atomWithSessionStorage } from './utils';

export const questionAtom = atomWithSessionStorage<IMessage[]>('questions', [{
  createdAt: new Date(),
  isFavorited: false,
  likeStatus: 0,
  content: '',
  messageId: '',
  role: '',
}])
