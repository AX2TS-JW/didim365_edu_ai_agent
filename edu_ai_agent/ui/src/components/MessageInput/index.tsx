import { Button, InputAdornment, TextField, Typography } from "@mui/material"
import { SendColorIcon, SendIcon } from "../Icons"

import styles from './MessageInput.module.scss';
import { useChat } from "@/hooks/useChat";
import { MyPromptModal } from "../MyPromptModal";

interface MessageInput {
  handleSendMessage: (message: string) => void;
}

export const MessageInput = (props: MessageInput) => {
  const {
    isOpen,
    message,
    handleMsgChange,
    onKeyDown,
    openPromptModal
  } = useChat();

  const { handleSendMessage } = props;
  const { container, textBox, sendIcon, options } = styles;

  return (
    <div className={container}>
      <div className={options}>
        {isOpen &&
          <MyPromptModal onClose={() => openPromptModal()} />
        }
        <Button onClick={openPromptModal}>
          <Typography>
            {'My prompt'}
          </Typography>
        </Button>
      </div>
      <div className={textBox}>
        <TextField
          fullWidth
          multiline
          minRows={1}
          maxRows={4}
          value={message}
          onChange={handleMsgChange}
          onKeyDown={onKeyDown}
          sx={{
            '& .MuiOutlinedInput-root': {
              padding: 0,
            },

            '& fieldset': {
              borderColor: 'rgba(0,0,0, 1)'
            },
            '& .Mui-focused fieldset': {
              borderColor: 'none',
              borderWidth: '10px'
            },
            '& .MuiInputAdornment-root': {
              alignSelf: 'flex-end',
              cursor: 'pointer'
            }
          }}
          placeholder='메시지를 입력하세요...'
          slotProps={{
            input: {
              endAdornment: (
                <InputAdornment position='end'>
                  <div className={sendIcon} onClick={() => handleSendMessage(message)}>
                    {message?.trim() === '' ? <SendIcon /> : <SendColorIcon />}
                  </div>
                </InputAdornment>
              )
            }
          }}
        />
      </div>
    </div>
  )
}