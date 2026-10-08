import { useEffect } from "react";
import { useMap } from "react-leaflet";

type Props = {
  center: [number, number];
  zoom: number;
};

export function MapFlyTo({ center, zoom }: Props) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}
