from controller.local_planner import LocalPlanner
import carla
import numpy as np


def convert_carla_image_to_array(image, converter=carla.ColorConverter.Raw):
    image.convert(converter)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def pp_local_planner(director, trajectory, sync_mode, render_display, display, hud, scene_img, ssd2d):
    def center2rear(node, wheelbase=2.850):
        theta, r = node[2] + np.pi, wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return tuple(node)
    wpts = [((t[0], -t[1], -t[2], t[3]), t[4] / 3.0) for t in trajectory]
    index = []
    for i, wp in enumerate(wpts[1:]):
        if wp[1] * wpts[i - 1][1] < 0:
            index.append(i)
    index.append(len(wpts))
    index.insert(0, 0)
    index = list(zip(index[:-1], index[1:]))
    segments = list(map(lambda x: wpts[x[0]:x[1]], index))
    local_planner = LocalPlanner(director.world)
    frames, threshold = 0, 500
    for i, seg in enumerate(segments):
        print('Segment {}/{} from {} to {}, type: {}'.format(
            i, len(segments), seg[0][0], seg[-1][0], np.sign(seg[0][1])))
        local_planner.set_waypoints_queue(seg)
        while local_planner.waypoints_queue:
            snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
            # front_view = ssd512.ssd_inference(ssd2d, convert_carla_image_to_array(front_view))
            front_view = back_view if director.player.get_control().reverse else front_view
            text = 'PP Controller Tracking the Parking Path Before the Next Re-planning, {:.1f} %'.format((frames+1)/threshold*100)
            render_display(display, hud, front_view, bird_view, scene_img, text=text, wait=0)
            tra = director.player.get_transform()
            vel = director.player.get_velocity()
            state = (tra.location.x, -tra.location.y, -np.radians(tra.rotation.yaw))
            state = director.state2transform(center2rear(list(state)), z=0.2)
            director.draw_arrow(state, carla.Color(r=0, b=255, g=0, a=255), z=3.2)
            director.draw_box(state, carla.Color(r=0, b=255, g=0, a=255), lift_time=-1.)
            local_planner.update((tra.location.x, tra.location.y, np.radians(tra.rotation.yaw)), (vel.x, vel.y))
            control = local_planner.run_step()
            if control:
                director.player.apply_control(control)
            director.move_spectator()
            frames += 1
            print('Parking pan Frame {}'.format(frames))
            if frames >= threshold:
                return
    control = director.player.get_control()
    control.steer = 0
    control.throttle = 0
    control.reverse = False
    control.hand_brake = True
    director.player.apply_control(control)
    snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
    # front_view = ssd512.ssd_inference(ssd2d, convert_carla_image_to_array(front_view))
    front_view = back_view if director.player.get_control().reverse else front_view
    render_display(display, hud, front_view, bird_view, scene_img, text='Parked', wait=0)


def pp_local_planner_without_replanning(director, trajectory, sync_mode, render_display, display, hud, scene_img, ssd2d):
    def center2rear(node, wheelbase=2.850):
        theta, r = node[2] + np.pi, wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return tuple(node)
    wpts = [((t[0], -t[1], -t[2], t[3]), t[4] / 3.0) for t in trajectory]
    index = []
    for i, wp in enumerate(wpts[1:]):
        if wp[1] * wpts[i - 1][1] < 0:
            index.append(i)
    index.append(len(wpts))
    index.insert(0, 0)
    index = list(zip(index[:-1], index[1:]))
    segments = list(map(lambda x: wpts[x[0]:x[1]], index))
    local_planner = LocalPlanner(director.world)
    passed_amount = 0
    for i, seg in enumerate(segments):
        print('Segment {}/{} from {} to {}, type: {}'.format(
            i, len(segments), seg[0][0], seg[-1][0], np.sign(seg[0][1])))
        local_planner.set_waypoints_queue(seg)
        passed_amount += len(seg)
        while local_planner.waypoints_queue:
            snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
            # front_view = ssd512.ssd_inference(ssd2d, convert_carla_image_to_array(front_view))
            front_view = back_view if director.player.get_control().reverse else front_view
            progress = (len(seg) - len(local_planner.waypoints_queue)) + (passed_amount - len(seg))
            text = 'PP Controller Tracking the Parking Path {:.1f} %'.format(progress / len(wpts) * 100)
            render_display(display, hud, front_view, bird_view, scene_img, text=text, wait=0)
            tra = director.player.get_transform()
            vel = director.player.get_velocity()
            state = (tra.location.x, -tra.location.y, -np.radians(tra.rotation.yaw))
            state = director.state2transform(center2rear(list(state)), z=0.2)
            director.draw_arrow(state, carla.Color(r=0, b=255, g=0, a=255), z=3.2)
            director.draw_box(state, carla.Color(r=0, b=255, g=0, a=255), lift_time=-1.)
            local_planner.update((tra.location.x, tra.location.y, np.radians(tra.rotation.yaw)), (vel.x, vel.y))
            control = local_planner.run_step()
            if control:
                director.player.apply_control(control)
            director.move_spectator()
    control = director.player.get_control()
    control.steer = 0
    control.throttle = 0
    control.reverse = False
    control.hand_brake = True
    director.player.apply_control(control)
    snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
    # front_view = ssd512.ssd_inference(ssd2d, convert_carla_image_to_array(front_view))
    front_view = back_view if director.player.get_control().reverse else front_view
    render_display(display, hud, front_view, bird_view, scene_img, text='Parked', wait=0)


def simple_local_planner(director, trajectory, sync_mode, render_display, display, hud, scene_img, ssd2d):
    def rear2center(node, wheelbase=2.850):
        theta, r = node[2], wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return tuple(node)

    def center2rear(node, wheelbase=2.850):
        theta, r = node[2] + np.pi, wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return tuple(node)
    tjs = [rear2center(list(t[:3])) for t in trajectory]
    tjt = [director.state2transform(t, z=director.player.get_transform().location.z) for t in tjs]
    tcs = [-t[3] for t in trajectory]
    sas = [np.degrees(np.arctan2(2.850 * tc, 1.)) / 70. for tc in tcs]
    vts = [t[4] for t in trajectory]
    wps = list(zip(tjt, sas, vts))
    director.player.set_simulate_physics(False)
    v0 = 0
    while wps:
        transform, steer, vt = wps.pop(0)
        director.player.set_transform(transform)
        control = director.player.get_control()
        control.steer = steer
        director.player.apply_control(control)
        tra = director.player.get_transform()
        state = (tra.location.x, -tra.location.y, -np.radians(tra.rotation.yaw))
        state = director.state2transform(center2rear(list(state)), z=0.27)
        director.draw_box(state, carla.Color(r=0, b=255, g=0, a=255), lift_time=-1.)
        snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
        text = 'PP Controller Tracking the Parking Path {:.1f} %'.format((1 - len(wps)/len(tjs)) * 100)
        front_view = back_view if v0 < -2 and vt < -2 else front_view
        v0 = vt
        render_display(display, hud, front_view, bird_view, scene_img, text=text, wait=0)

    director.player.set_simulate_physics(True)
    snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
    render_display(display, hud, front_view, bird_view, scene_img, text='Parked', wait=0)


def simple_local_planner_with_replanning(
        director, trajectory, sync_mode, render_display, display, hud, scene_img, ssd2d):
    def rear2center(node, wheelbase=2.850):
        theta, r = node[2], wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return tuple(node)
    tjs = [rear2center(list(t[:3])) for t in trajectory]
    tjt = [director.state2transform(t, z=director.player.get_transform().location.z) for t in tjs]
    tcs = [-t[3] for t in trajectory]
    sas = [np.degrees(np.arctan2(2.850 * tc, 1.)) / 70. for tc in tcs]
    vts = [t[4] for t in trajectory]
    wps = list(zip(tjt, sas, vts))
    director.player.set_simulate_physics(False)
    frames, threshold = 0, 5
    while wps:
        transform, steer, vt = wps.pop(0)
        director.player.set_transform(transform)
        control = director.player.get_control()
        control.steer = steer
        director.player.apply_control(control)
        snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
        text = 'PP Controller Tracking the Parking Path Before the Next Re-planning, {:.1f} %'.format((frames+1)/threshold*100)
        front_view = back_view if vt < 0 else front_view
        render_display(display, hud, front_view, bird_view, scene_img, text=text, wait=0)
        frames += 1
        if frames >= threshold:
            return

    director.player.set_simulate_physics(True)
    snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
    render_display(display, hud, front_view, bird_view, scene_img, text='Parked', wait=0)
