import { Outlet } from 'react-router-dom';
import {
  Box,
} from '@mui/material';

import styles from './Layout.module.scss';
import { SubMenu } from './Menu/SubMenu';
import { Menu } from './Menu';

const { container } = styles;

export const Layout = () => {
  return (
    <Box className={container}>
      <Menu>
        <SubMenu />
      </Menu>

      <Box component='main' sx={{
        position: 'relative',
        overflow: 'hidden',
        flexGrow: 1
      }}>
        <Outlet />
      </Box>
    </Box>
  )
}