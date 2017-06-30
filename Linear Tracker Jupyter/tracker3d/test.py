""" Sup B&R.
    It's your god bud, Dan. I got some potential solutions for you.
    I didn't check my work.
    Maybe they can serve as inspiration for an even better solution.
    Everything's in python syntax, but it looks enough like psuedo code
    that it probably won't matter.
    norm2(v) - Take the 2 norm of a vector
    dot(v, w) Dot the v and w vectors.
    normalize(v) - Return a vector with v's direction, but magnitude of 1.
    project(v, n) - Project v onto the vector with norm of n.
"""

def bounce(p_pos, p_vel, wall_pos, wall_norm, bounce_friction):
    """ Have the player bounce off the wall
    Arguments:
        p_pos (3D vector):
            The player's current position measured in units.
        p_vel (3D vector):
            The player's current velocity measured in units per frame.
        wall_pos (3D vector):
            Position where the player will hit the wall without adjustment.
        wall_norm (3D vector):
            The wall's normal vector.
        bounce_friction (float):
            A multiplier (0, 1] for how fast a bounced player should go
            relative to their previous speed.
    Returns: (3D vector)
        The recommended new player vector. 
    """
    if norm2(p_pos - wall_pos) < p_vel:
        # Uh oh! Next frame, the player and the wall will collide!
        # Let's just bounce the player off the wall. Most math libraries
        # have a vector reflection function, so use that. Here's the pure
        # math for a reflection, though.
        new_vel = p_vel - 2 * dot(dot(p_vel, wall_norm), wall_norm)
        return bounce_friction * new_vel
    else:
        # The player will not collide with the wall next frame. Do nothing.
        return p_vel

def glide(p_pos, p_vel, wall_pos, wall_norm, glide_friction):
    """ Have the player glide along the wall.
    Arguments:
        p_pos (3D vector):
            The player's current position measured in units.
        p_vel (3D vector):
            The player's current velocity measured in units per frame.
        wall_pos (3D vector):
            Position where the player will hit the wall without adjustment.
        wall_norm (3D vector):
            The wall's normal vector.
        glide_friction (float):
            A multiplier (0, 1] for how fast a glided player should go
            relative to their previous speed.
    Returns: (3D vector)
        The recommended new player vector. 
    """
    if norm2(p_pos - wall_pos) < p_vel:
        # Most math libraries include a projection function.
        new_vec = normalize(project(p_vel, wall_norm)) * norm2(p_vel)
        return new_vec * glide_friction
    else:
        return p_vel

def stop(p_pos, p_vel, wall_pos, wall_norm, stop_friction):
    """ Have the player gradually come to a halt.
    Arguments:
        p_pos (3D vector):
            The player's current position measured in units.
        p_vel (3D vector):
            The player's current velocity measured in units per frame.
        wall_pos (3D vector):
            Position where the player will hit the wall without adjustment.
        wall_norm (3D vector):
            The wall's normal vector.
        stop_friction (float):
            A multiplier (0, 1] for how fast a stopping player should go
            relative to their previous speed.
    Returns: (3D vector)
        The recommended new player vector. 
    """
    if norm2(p_pos - wall_pos) < p_vel:
        new_vel = 2norm(p_pos - wall_pos)
        return stop_friction * new_vel
    else:
        return p_vel
    
def deflect(p_pos, p_vel, wall_pos, wall_norm, deflect_friction):
    if norm2(p_pos - wall_pos) > p_vel:
        return p_vel
    
    
/* Get the new velocity vector after a deflection
 * pp - Player's current position (units)
 * pv - Player's current velocity (units per frame)
 * wp - Where the player will collide with a wall if no changes to pv occur
 * wn - The wall's normal vector
 * f  - The friction multiplier. If a deflection occurs, the speed is multiplied by this.
 */
public FVector deflect (FVector pp, FVector pv, FVector wp, FVector wn, double f) {
    if (FVector::Dist(pp, wp) > FVector::Size(pv)) {
        return pv
    }    
}