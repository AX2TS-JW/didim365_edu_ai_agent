import CodeMirror, { EditorView } from '@uiw/react-codemirror';
import { sql } from '@codemirror/lang-sql';
import styles from './CodeEditor.module.scss';
import { aura } from '@uiw/codemirror-theme-aura';
import { Box, Typography } from '@mui/material';
import { ArrowDownIcon, CopyIcon } from '../Icons';
import { useState } from 'react';

interface CodeEditorType {
  code: string;
}

export const CodeEditor = (props: CodeEditorType) => {
  const { code } = props;
  const { container, header, title, icon } = styles;
  const [isCollapsed, setIsCollapsed] = useState(false);

  const custom = EditorView.theme({
    "&": {
      borderRadius: '12px'
    },
    ".cm-scroller": {
      borderRadius: "12px",
      "&::-webkit-scrollbar": {
        display: "none"
      },
    },
    ".cm-gutters": {
      borderRadius: "12px 0 0 12px",
      border: "none",
    },
    '.cm-foldGutter span[title="Unfold line"]': {
      position: 'relative',
      top: '-2px'
    },
    '.cm-foldGutter span[title="Fold line"]': {
      position: 'relative',
      top: '-9px'
    },
    ".cm-content": {
      fontSize: '12px',
      // fontFamily: "'Fira Code', monospace",
    },
  })

  const handleCopyCode = async (e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(code)
    } catch (err) {
      console.log('복사 실패:', err);
    }
  };

  const handleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  }

  return (
    <div className={container}>
      <div className={header} onClick={handleCollapse}>
        <div className={title}>
          <Typography variant='h6'>
            {'코드'}
          </Typography>
          <ArrowDownIcon></ArrowDownIcon>
        </div>
        <Box
          className={icon} onClick={handleCopyCode}
          sx={{
            display: 'flex',
          }}
        >
          <CopyIcon color={'#737373'}></CopyIcon>
        </Box>
      </div>
      {isCollapsed && (
        <div style={{ margin: '0px 20px 16px 20px' }}>
          <CodeMirror
            value={code}
            height='180px'
            theme={aura}
            extensions={[sql(), custom]}
          />
        </div>
      )}
    </div>
  )
}