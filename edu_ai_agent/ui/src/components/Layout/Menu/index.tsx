import { PropsWithChildren } from "react";
import { Box, Drawer, Typography } from "@mui/material";
import styles from './Menu.module.scss';

import { HomeColorIcon, HomeIcon, LogoIcon, MessageColorIcon, MessageIcon, SettingColorIcon, SettingIcon } from "@/components/Icons";
import { useLocation, useNavigate } from "react-router-dom";
import { useAtomValue } from "jotai";
import { questionAtom } from "@/store/question";


export const Menu: React.FC<PropsWithChildren> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const question = useAtomValue(questionAtom);
  const { container, logo, wrap, menu, } = styles;

  const menuItems = [
    {
      id: 'dashboard',
      icon: <HomeIcon color={'#737373'} />,
      activeIcon: <HomeColorIcon />,
      label: '대시보드',
      paths: ['/dashboard']
    },
    {
      id: 'chat',
      icon: <MessageIcon color={'#737373'} />,
      activeIcon: <MessageColorIcon />,
      label: 'Chat',
      paths: ['/', '/chat']
    },
    {
      id: 'setting',
      icon: <SettingIcon color={'#737373'} />,
      activeIcon: <SettingColorIcon />,
      label: '관리',
      paths: ['/setting']
    }
  ];

  const isActiveIcon = (item) => {
    return item.paths.includes(location.pathname)
  }

  const handleRoutePage = (route: string) => {
    navigate(route);
    if (route == 'chat') {
      navigate('')
    } else {
      navigate(route);
    }
  };

  return (
    <>
      <Drawer
        variant="permanent"
        className={container}
        sx={{
          width: 64,
          '& .MuiDrawer-paper': {
            width: 64,
            boxSizing: 'border-box',
            bgcolor: '#ffffff',
            zIndex: 1300,
            //boxShadow: '0 4px 8px rgba(0,0,0, .08)'
          }
        }}>

        <div className={logo} onClick={() => handleRoutePage('')}>
          <LogoIcon />
        </div>

        <Box className={wrap}>
          <ul className={menu}>
            {menuItems.map((item) => (
              <li key={item.id} onClick={() => handleRoutePage(item.id)}>
                <Box
                  sx={{
                    borderRadius: '8px',
                    backgroundColor: isActiveIcon(item) ? '#E1E5ff' : 'inherit'
                  }}
                >
                  {isActiveIcon(item) ? item.activeIcon : item.icon}
                </Box>
                <Typography fontSize={14}>
                  {item.label}
                </Typography>
              </li>
            ))}
          </ul>
        </Box>
      </Drawer>
      {children}
    </>
  )
}