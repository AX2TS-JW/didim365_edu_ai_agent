import { atom } from 'jotai';

import { IMessage } from '../types/chatVM';
import { atomWithSessionStorage } from './utils';

export const answerAtom = atomWithSessionStorage<IMessage[]>('answers', [{
	createdAt: new Date(),
	isFavorited: false,
	likeStatus: 0,
	content: '',
	messageId: '',
	role: '',
	metadata: {
		sql: {},
		data: {},
		chart: {}
	},
}])
