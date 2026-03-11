# Agent Education Template UI

React 기반의 Agent 교육용 프론트엔드 템플릿입니다.

## 기술 스택

- React 18
- Vite
- pnpm

## 교육생 환경 준비 및 실행 가이드

이 프로젝트를 처음 설정하고 실행하기 위해 아래 단계에 따라 환경을 구성해주세요.

### 1. 패키지 매니저 설치 (pnpm)
이 프로젝트는 패키지 관리를 위해 `pnpm`을 사용합니다. 아직 설치되어 있지 않다면 아래 명령어로 `pnpm`을 전역으로 설치해주세요. (Node.js는 이미 설치되어 있어야 합니다.)

```bash
npm install -g pnpm
```

### 2. 프로젝트 의존성 설치
프로젝트 루트 디렉토리(이 `README.md` 파일이 있는 위치)에서 다음 명령어를 실행하여 필요한 패키지들을 설치합니다.

```bash
pnpm install
```

### 3. 개발 서버 실행
설치가 완료되면 다음 명령어로 로컬 개발 서버를 시작합니다.

```bash
pnpm dev
```
명령어를 실행하면 보통 `http://localhost:5173` 으로 서버가 구동됩니다. 터미널에 표시된 주소를 브라우저에서 열어 확인하세요.

### (참고) 프로젝트 빌드
교육 과정 중 결과물을 빌드하여 배포하는 단계가 있다면 아래 명령어를 사용합니다.

```bash
pnpm build
```

## 환경 변수

`.env` 파일을 생성하고 다음 변수를 설정하세요:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## 프로젝트 구조

```text
ui/
├── src/
│   ├── components/    # 재사용 가능한 컴포넌트
│   ├── hook/          # 훅 및 이벤트 모음
│   ├── lib/           # JS 라이브러리 설정
│   ├── pages/         # 페이지 컴포넌트
│   ├── services/      # API 호출 및 비즈니스 로직
│   ├── store/         # 상태 관리 및 middleware 모음
│   ├── utils/         # 유틸리티 함수
│   ├── App.jsx        # 메인 앱 컴포넌트
│   ├── App.css        # 앱 스타일
│   ├── main.jsx       # 진입점
│   └── index.css      # 전역 스타일
├── public/            # 정적 파일
├── index.html         # HTML 템플릿
├── vite.config.js     # Vite 설정
└── package.json       # 프로젝트 메타데이터
```

### 디버깅 실행 순서
1. 터미널을 열고 `pnpm dev` 명령어로 개발 서버를 실행시킵니다.
2. VSCode 좌측 메뉴의 **Run and Debug** (또는 `Cmd+Shift+D` / `Ctrl+Shift+D`) 패널을 엽니다.
3. 상단 설정에서 **"Launch Chrome against localhost"**를 선택하고 실행(`F5`) 버튼을 클릭합니다.
4. VSCode 상의 소스 코드에 중단점(Breakpoint)을 설정한 후, 새로 열린 브라우저 창에서 동작을 테스트하여 디버깅을 진행할 수 있습니다.
