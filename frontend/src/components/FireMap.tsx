import { CircleMarker, MapContainer, TileLayer, Popup } from "react-leaflet";
import type { Frame, FrameCell } from "../api";
import { MapFlyTo } from "./MapFlyTo";
import "leaflet/dist/leaflet.css";

type Layers = {
  spikes: boolean;
  membrane: boolean;
  detect: boolean;
  tracks: boolean;
  forecast: boolean;
};

type Props = {
  frame: Frame | null;
  layers: Layers;
  center: [number, number];
  zoom: number;
};

function visible(cell: FrameCell, layers: Layers) {
  const interesting = cell.spike || cell.v >= 0.25 || cell.track > 0;
  const forecastHit = layers.forecast && cell.forecast >= 0.55 && cell.v >= 0.2;
  return interesting || forecastHit;
}

function colorFor(cell: FrameCell, layers: Layers) {
  const interesting = cell.spike || cell.v >= 0.25 || cell.track > 0;
  const forecastHit = layers.forecast && cell.forecast >= 0.55 && cell.v >= 0.2;
  let color = "#b85c38";
  if (layers.membrane && cell.v >= 0.25) color = "#8b5a2b";
  if (forecastHit) color = "#2f6f6a";
  if (layers.detect && cell.det >= 0.7 && interesting) color = "#8c2f2f";
  if (layers.tracks && cell.track) color = "#3d4c7a";
  if (layers.spikes && cell.spike) color = "#c4552a";
  return color;
}

export function FireMap({ frame, layers, center, zoom }: Props) {
  const cells = (frame?.cells || []).filter((cell) => visible(cell, layers));
  return (
    <MapContainer center={center} zoom={zoom} className="map" zoomControl>
      <MapFlyTo center={center} zoom={zoom} />
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {cells.map((cell, index) => (
        <CircleMarker
          key={`${cell.lat}-${cell.lon}-${index}`}
          center={[cell.lat, cell.lon]}
          radius={layers.membrane ? 3 + Math.min(cell.v, 3) * 2 : 4}
          pathOptions={{
            color: colorFor(cell, layers),
            weight: 1,
            fillOpacity: 0.8,
          }}
        >
          <Popup>
            membrane {cell.v}
            <br />
            detection {cell.det}
            <br />
            forecast {cell.forecast}
            <br />
            track {cell.track}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
