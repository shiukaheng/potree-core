import { PlaneGeometry, Points, ShaderMaterial, Vector3 } from "three";

export function createAnimatedPointPlane() {
    // Step 1: Create a plane geometry
    const planeGeometry = new PlaneGeometry(100, 70, 300, 300);
    planeGeometry.rotateZ(0.42);
    // Step 2: Create a custom shader material
    const pointShaderMaterial = new ShaderMaterial({
        vertexShader: `
            uniform float uTime;
            uniform vec3 wind_vector;
            uniform float wind_scale;
            uniform vec3 displacement_vector;
            uniform float sigmoid_alpha;
            uniform float sigmoid_beta;
            varying vec3 pointColor;
                
                        
            // Simplex noise

            vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
            float permute(float x){return floor(mod(((x*34.0)+1.0)*x, 289.0));}
            vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
            float taylorInvSqrt(float r){return 1.79284291400159 - 0.85373472095314 * r;}

            vec4 grad4(float j, vec4 ip){
            const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
            vec4 p,s;

            p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
            p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
            s = vec4(lessThan(p, vec4(0.0)));
            p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www; 

            return p;
            }

            float snoise(vec4 v){
            const vec4 C = vec4(0.138196601125010504,  // (5 - sqrt(5))/20  G4
                                0.309016994374947451,  // (sqrt(5) - 1)/4   F4
                                0.0,
                                0.0);
            // First corner
            vec4 i  = floor(v + dot(v, C.yyyy) );
            vec4 x0 = v -   i + dot(i, C.xxxx);

            // Other corners

            // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
            vec4 i0;
            vec3 isX = step( x0.yzw, x0.xxx );
            vec3 isYZ = step( x0.zww, x0.yyz );
            i0.x = isX.x + isX.y + isX.z;
            i0.yzw = 1.0 - isX;
            i0.y += isYZ.x + isYZ.y;
            i0.zw += 1.0 - isYZ.xy;
            i0.z += isYZ.z;
            i0.w += 1.0 - isYZ.z;

            // i0 now contains the unique values 0,1,2,3 in each channel
            vec4 i3 = clamp( i0, 0.0, 1.0 );
            vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
            vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

            vec4 x1 = x0 - i1 + C.xxxx;
            vec4 x2 = x0 - i2 + C.yyyy;
            vec4 x3 = x0 - i3 + C.zzzz;
            vec4 x4 = x0 + C.wwww;

            // Permutations
            i = mod(i, 289.0); 
            float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
            vec4 j1 = permute( permute( permute( permute (
                        i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
                    + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
                    + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
                    + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));
                    
            // Gradients
            // ( 7*7*6 points uniformly over a cube, mapped onto a 4-octahedron.)
            // 7*7*6 = 294, which is close to the ring size 17*17 = 289.

            vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

            vec4 p0 = grad4(j0,   ip);
            vec4 p1 = grad4(j1.x, ip);
            vec4 p2 = grad4(j1.y, ip);
            vec4 p3 = grad4(j1.z, ip);
            vec4 p4 = grad4(j1.w, ip);

            // Normalise gradients
            vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
            p0 *= norm.x;
            p1 *= norm.y;
            p2 *= norm.z;
            p3 *= norm.w;
            p4 *= taylorInvSqrt(dot(p4,p4));

            // Mix contributions from the five corners
            vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
            vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
            m0 = m0 * m0;
            m1 = m1 * m1;
            return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
                            + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;
            }
    
            vec3 calculateDistortion(vec3 position, float intensity, float time, vec3 wind_vector, float wind_scale, vec3 displacement_vector, float sigmoid_alpha, float sigmoid_beta) {
                float scaledConfidence = 1.0 / (1.0 + exp(sigmoid_alpha * (intensity - sigmoid_beta)));
                float wind_vector_length = length(wind_vector);
                vec3 noiseInput = position + time * wind_vector;
    
                float noise = snoise(vec4(noiseInput * wind_scale, time * wind_vector_length / 2.0)) +
                              snoise(vec4(noiseInput * wind_scale * 5.0, time * wind_vector_length / 2.0)) +
                              snoise(vec4(noiseInput * wind_scale * 50.0, time * wind_vector_length / 2.0)) / 3.0;
    
                float offset = scaledConfidence * abs(noise);
                return displacement_vector * offset;
            }
    
            void main() {
                vec3 transformedPosition = position;
                float intensity = 1.0; // For now, we'll keep intensity constant
    
                vec3 distortion = calculateDistortion(transformedPosition, intensity, uTime, wind_vector, wind_scale, displacement_vector, sigmoid_alpha, sigmoid_beta);
                transformedPosition += distortion; // Apply distortion

                pointColor = vec3(0.756, 0.839, 0.819); // Set the color of the point
    
                gl_Position = projectionMatrix * modelViewMatrix * vec4(transformedPosition, 1.0);

                // Exponential fog (black)
                vec3 fogColor = vec3(1, 1, 1);
                float fogDensity = 0.03;
                float z = gl_Position.w;
                float d = z * fogDensity;
                float fogFactor = 1.0 - exp2(-d * d);
                pointColor = mix(pointColor, fogColor, fogFactor);

                gl_PointSize = 3.0; // Set the size of the points

                
            }
        `,
        fragmentShader: `
            varying vec3 pointColor;

            void main() {
                gl_FragColor = vec4(pointColor, 1.0); // Set the color of the point
            }
        `,
        uniforms: {
            uTime: { value: 0.0 }, // Time uniform to animate points
            wind_vector: { value: new Vector3(1.0, 0.0, 0.0) }, // Wind vector
            wind_scale: { value: 0.5 }, // Wind scale factor
            displacement_vector: { value: new Vector3(0.0, 0.0, 2.0) }, // Displacement direction
            sigmoid_alpha: { value: 1.0 }, // Sigmoid alpha for distortion scaling
            sigmoid_beta: { value: 0.5 }  // Sigmoid beta for distortion scaling
        }
    });

    // Step 3: Create a Points object from the plane geometry
    const pointPlane = new Points(planeGeometry, pointShaderMaterial);

    return pointPlane;
}