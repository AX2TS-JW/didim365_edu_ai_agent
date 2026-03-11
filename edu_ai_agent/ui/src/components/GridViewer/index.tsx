import styles from './GridViewer.module.scss';
import { useState } from 'react';
import { Typography } from '@mui/material';
import { ArrowDownIcon } from '../Icons';
import { Table } from './Table';

interface GridViewerProps {
  data: any;
  collapse: boolean,
}

export const GridViewer = (props: GridViewerProps) => {
  const { data, collapse } = props;
  const { container, header, title, theme, icon } = styles;
  const [isCollapsed, setIsCollapsed] = useState(collapse);

  const handleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  }

  return (
    <div className={container}>
      <div className={header} onClick={handleCollapse}>
        <div className={title}>
          <Typography
            variant='h6'
          >
            {'데이터(표)'}
          </Typography>
          <ArrowDownIcon></ArrowDownIcon>
        </div>
      </div>

      {isCollapsed && (
        <div style={{ margin: '0px 20px 16px 20px' }}>
          <Table
            theme={theme}
            config={data}
          />
        </div>
      )}
    </div>
  )
}