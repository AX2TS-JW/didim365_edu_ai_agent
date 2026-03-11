import { createBrowserRouter, RouteObject } from "react-router-dom";
import { routeConfig } from "./route.config";

export const router = createBrowserRouter(routeConfig as unknown as RouteObject[]);