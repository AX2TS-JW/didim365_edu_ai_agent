import { useEffect, useRef, useMemo } from "react"
import Grid from '@highcharts/grid-lite/es-modules/masters/grid-lite.src';
import '@highcharts/grid-lite/css/grid.css';
import DataTable from "@highcharts/grid-lite/es-modules/Data/DataTable";

interface GridTable {
	config: Grid.Options;
	theme: string;
}

// 숫자인지 확인하는 함수
const isNumeric = (value: any): boolean => {
	if (typeof value === 'number') {
		return !isNaN(value) && isFinite(value);
	}
	if (typeof value === 'string') {
		const trimmed = value.trim();
		const withoutCommas = trimmed.replace(/,/g, '');

		return withoutCommas !== '' && /^-?\d+(\.\d+)?$/.test(withoutCommas);
	}

	return false;
};
// 데이터가 비어있는지 확인
const isEmpty = (value: any): boolean => {
	return value === null || value === undefined || value === '' || (typeof value === 'string' && value.trim() === '');
};

export const Table = (props: GridTable) => {
	const { config, theme } = props;
	const containerRef = useRef<HTMLDivElement>(null);

	// 데이터 변환 및 처리
	const processedConfig = useMemo(() => {
		if (!config.dataTable || !config.dataTable.columns) {
			return config;
		}

		const { columns } = config.dataTable as DataTable;
		const columnNames = Object.keys(columns);

		if (columnNames.length === 0) {
			return config;
		}

		// 행 수 확인
		const rowCount = columns[columnNames[0]]?.length || 0;

		// 컬럼 정의 생성 (첫 번째 컬럼은 56px 고정)
		const gridColumns = columnNames.map((colName, index) => {
			const firstValue = columns[colName]?.find(v => !isEmpty(v));
			const isNumericColumn = isNumeric(firstValue);

			return {
				id: colName,
				title: colName,
				cells: {
					className: isNumericColumn ? 'hcg-right' : 'hcg-left',
				},
				header: {
					className: isNumericColumn ? 'hcg-right' : 'hcg-left',
				}
			};
		});

		// 데이터 행 생성 (빈 값은 "-"로 변환)
		const rows = Array.from({ length: rowCount }, (_, rowIndex) => {
			return columnNames.map((colName) => {
				const value = columns[colName]?.[rowIndex];
				return isEmpty(value) ? '-' : value;
			});
		});

		return {
			...config,
			columns: gridColumns,
			data: rows
		};
	}, [config]);

	useEffect(() => {
		if (containerRef.current) {
			// 기존 그리드 제거
			containerRef.current.innerHTML = '';

			// 그리드 생성
			Grid.grid(containerRef.current, processedConfig as any);
		}
	}, [processedConfig]);

	return <div ref={containerRef} className={theme} />
}