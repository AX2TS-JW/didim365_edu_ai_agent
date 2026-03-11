import { Typography } from '@mui/material';
import style from './MyPropmtModal.module.scss';
import { PlusIcon } from '../Icons';
import { useChat } from '@/hooks/useChat';
import { useEffect, useRef } from 'react';
import { useAtomValue } from 'jotai';
import { promptAtom } from '@/store/prompts';

interface MyPromptModalType {
  onClose: () => void;
}

export const MyPromptModal = (props: MyPromptModalType) => {
  const { onClose } = props;

  const {
    handleUsePrompt
  } = useChat();

  const { container, header } = style;
  const propmt = useAtomValue(promptAtom);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [onClose])

  return (
    <div className={container} ref={modalRef}>
      <div className={header}>
        <Typography variant='h6'>
          {'My prompt'}
        </Typography>
        <PlusIcon color={'#404040'}></PlusIcon>
      </div>
      <ul>
        {propmt.map((item, i) => {
          return (
            <li key={i} onClick={() => handleUsePrompt(item)}>
              <Typography>
                {item}
              </Typography>
            </li>
          )
        })}
      </ul>
    </div>
  )
}