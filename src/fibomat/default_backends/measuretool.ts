// Measure tool for bokeh plots: drag with the mouse to measure the distance and the angle between two points.
//
// This file is compiled by bokeh (see `scripts/build_measuretool.py`, which creates `measuretool.js`). Never edit
// `measuretool.js` by hand.
import {GestureTool, GestureToolView} from "models/tools/gestures/gesture_tool"
import {ColumnDataSource} from "models/sources/column_data_source"
import {Label} from "models/annotations/label"
import type {PanEvent, TapEvent} from "core/ui_events"
import * as p from "core/properties"

export class MeasureToolView extends GestureToolView {
  declare model: MeasureTool

  protected start: [number, number] | null = null

  protected _to_data(sx: number, sy: number): [number, number] | null {
    const {frame} = this.plot_view
    if (!frame.bbox.contains(sx, sy))
      return null
    return [frame.x_scale.invert(sx), frame.y_scale.invert(sy)]
  }

  protected _reset(): void {
    this.start = null
    this.model.source.data = {x: [], y: []}
    this.model.label.text = ""
  }

  override _pan_start(ev: PanEvent): void {
    this.start = this._to_data(ev.sx, ev.sy)
  }

  override _pan(ev: PanEvent): void {
    if (this.start == null)
      return

    const end = this._to_data(ev.sx, ev.sy)
    if (end == null)
      return

    const [x0, y0] = this.start
    const [x1, y1] = end
    const dx = x1 - x0
    const dy = y1 - y0

    const distance = Math.sqrt(dx * dx + dy * dy)
    const angle_rad = Math.atan2(dy, dx)
    const angle_deg = angle_rad * 180 / Math.PI
    const unit = this.model.measure_unit == "" ? "" : " " + this.model.measure_unit

    this.model.source.data = {x: [x0, x1], y: [y0, y1]}
    this.model.label.text =
      "distance = " + distance.toFixed(3) + unit +
      ", angle = " + angle_deg.toFixed(2) + "° (" + angle_rad.toFixed(2) + " rad)"
  }

  override _pan_end(_ev: PanEvent): void {
    this.start = null
  }

  // a click removes the measurement
  override _tap(_ev: TapEvent): void {
    this._reset()
  }

  override deactivate(): void {
    super.deactivate()
    this._reset()
  }
}

export namespace MeasureTool {
  export type Attrs = p.AttrsOf<Props>

  export type Props = GestureTool.Props & {
    source: p.Property<ColumnDataSource>
    label: p.Property<Label>
    measure_unit: p.Property<string>
  }
}

export interface MeasureTool extends MeasureTool.Attrs {}

export class MeasureTool extends GestureTool {
  declare properties: MeasureTool.Props
  declare __view_type__: MeasureToolView

  constructor(attrs?: Partial<MeasureTool.Attrs>) {
    super(attrs)
  }

  override tool_name = "Measure"
  override tool_icon = "bk-tool-icon-crosshair"
  override event_type = ["pan" as "pan", "tap" as "tap"]
  override default_order = 12

  static {
    this.prototype.default_view = MeasureToolView

    this.define<MeasureTool.Props>(({Ref, Str}) => ({
      source: [Ref(ColumnDataSource)],
      label: [Ref(Label)],
      measure_unit: [Str, ""],
    }))
  }
}
