import { AmbientLight, BoxGeometry, Color, Euler, Group, Matrix4, Mesh, MeshBasicMaterial, PerspectiveCamera, Raycaster, Scene, SphereGeometry, Vector2, Vector3, WebGLRenderer } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';
import { PointCloudOctree, Potree } from '../source';
import { PlaneGeometry, Points, ShaderMaterial, BufferGeometry, Float32BufferAttribute } from 'three';
import { createAnimatedPointPlane } from './animatedPlane';
import * as THREE from 'three';

document.body.onload = function() {
    const potree = new Potree();
	potree.pointBudget = 500000
    let pointClouds: { [key: string]: PointCloudOctree } = {};

    // three.js
    const scene = new Scene();
	// scene.background = new Color(0xc1d6d1);
    const camera = new PerspectiveCamera(60, 1, 0.1, 1000);
	
	// Apply background color using large double-sided cube
	const geometry = new BoxGeometry(100, 100, 100);
	const material = new MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
	const cube = new Mesh(geometry, material);
	scene.add(cube);

	// Add style for body so it would be black
	document.body.style.backgroundColor = 'black';
    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0px';
    canvas.style.left = '0px';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    document.body.appendChild(canvas);

    const renderer = new WebGLRenderer({
        canvas: canvas,
        alpha: true,
        logarithmicDepthBuffer: false,
        precision: 'highp',
        premultipliedAlpha: true,
        antialias: true,
        preserveDrawingBuffer: false,
        powerPreference: 'high-performance',
    });


    // Enable XR
    renderer.xr.enabled = true;
    scene.add(new AmbientLight(0xffffff));
    camera.position.z = 0;

    // Load each point cloud individually and store it in the pointClouds object
    loadPointCloud('boat', 'jumbo/boat/');
    loadPointCloud('dock_1', 'jumbo/dock_1/');
    loadPointCloud('floating_ring', 'jumbo/floating_ring/');
    loadPointCloud('kitchen', 'jumbo/kitchen/');
    loadPointCloud('platform', 'jumbo/platform/');
    loadPointCloud('sign_r', 'jumbo/sign_r/');
    loadPointCloud('boat_2', 'jumbo/boat_2/');
    loadPointCloud('dock_2', 'jumbo/dock_2/');
    loadPointCloud('jumbo', 'jumbo/jumbo/');
    loadPointCloud('misc', 'jumbo/misc/');
    loadPointCloud('sign_l', 'jumbo/sign_l/');
    loadPointCloud('tai_pak', 'jumbo/tai_pak/');

    function loadPointCloud(name: string, baseUrl: string, position?: Vector3, rotation?: Euler, scale?: Vector3) {
        potree.loadPointCloud('metadata.json', url => `${baseUrl}${url}`).then(function(pco: PointCloudOctree) {
            pco.material.size = 1.0;
            pco.material.shape = 0;
            pco.material.inputColorEncoding = 1;
            pco.material.outputColorEncoding = 1;

            if (position) { pco.position.copy(position); }
            if (rotation) { pco.rotation.copy(rotation); }
            if (scale) { pco.scale.copy(scale); }

            console.log(`Pointcloud ${name} loaded`, pco);
            pco.showBoundingBox = false;

            const box = pco.pcoGeometry.boundingBox;
            const size = box.getSize(new Vector3());

            // Store each point cloud in the pointClouds object with its corresponding name
            pointClouds[name] = pco;

			const group = new Group();
			group.add(pco);

			scene.add(group);
			scene.rotation.set(-Math.PI/2, 0, -Math.PI/2 + 0.05);
			scene.position.set(8, 0, 5);
        });
    }

    function addToUpdater(pco: PointCloudOctree): void {
        // Add to point clouds updater
    }

	// Function to create a plane rendered as points
	

	const pointPlane = createAnimatedPointPlane();
	scene.add(pointPlane);

    function unload(): void {
        Object.keys(pointClouds).forEach(key => {
            const pco = pointClouds[key];
            scene.remove(pco);
            pco.dispose();
        });

        pointClouds = {};
    }

    // WebXR setup
    document.body.appendChild(VRButton.createButton(renderer));

    function animate() {
        renderer.setAnimationLoop(render);
    }

	const startTime = performance.now();

    function render() {
		// Update time in the shader
		const elapsedTime = (performance.now() - startTime) / 1000;
		pointPlane.material.uniforms.uTime.value = elapsedTime;

        // Update each point cloud independently if needed
        Object.keys(pointClouds).forEach(key => {
            const pco = pointClouds[key];
			if (key === 'boat') {
				pco.rotation.x = Math.sin(Date.now() / 1000 + 50) / 50;
				pco.rotation.y = Math.cos(Date.now() / 1000 + 50) / 50;
				pco.rotation.z = Math.cos(Date.now() / 1000 + 30) / 50;
			} else if (key === 'boat_2') {
				pco.rotation.x = Math.sin(Date.now() / 1000 + 37) / 50;
				pco.rotation.y = Math.cos(Date.now() / 1000 + 50) / 50;
				pco.rotation.z = Math.cos(Date.now() / 1000 + 23) / 50;
			} else if (key === 'kitchen') {
				pco.rotation.x = Math.sin(Date.now() / 5000 + 76) / 60;
				pco.rotation.y = Math.cos(Date.now() / 5000 + 50) / 60;
				pco.rotation.z = Math.cos(Date.now() / 4000 + 21) / 50;
			} else if (key === 'sign_l') {
				pco.rotation.z = Math.sin(Date.now() / 4000 + 50) / 200;
				pco.rotation.x = Math.cos(Date.now() / 4000 + 50) / 500;
				pco.rotation.y = Math.cos(Date.now() / 4000 + 50) / 500;
			} else if (key === 'sign_r') { // Random rotation, different from above. Needs to be different for every object from this point on. e.g. 1645.8 > 1000
				pco.rotation.z = Math.sin(Date.now() / 4000 + 389) / 200;
				pco.rotation.x = Math.cos(Date.now() / 4000 + 389) / 500;
				pco.rotation.y = Math.cos(Date.now() / 4000 + 389) / 500;
			} else if (key === 'platform') {
				pco.rotation.x = Math.sin(Date.now() / 2000 + 50) / 70;
				pco.rotation.y = Math.cos(Date.now() / 2000 + 50) / 70;
				pco.rotation.z = Math.cos(Date.now() / 2000 + 50) / 100;
			} else if (key === 'tai_pak') {
				pco.rotation.x = Math.sin(Date.now() / 4000 + 50) / 400 + Math.sin(Date.now() / 2000 + 100) / 600;
				pco.rotation.y = Math.cos(Date.now() / 4000 + 50) / 400 + Math.cos(Date.now() / 2000 + 200) / 600;
				pco.rotation.z = Math.cos(Date.now() / 4000 + 50) / 400 + Math.sin(Date.now() / 2000 + 300) / 600;
			} else if (key === 'misc') {
				pco.rotation.x = Math.sin(Date.now() / 2000 + 350) / 70;
				pco.rotation.y = Math.cos(Date.now() / 2000 + 10) / 70;
				pco.rotation.z = Math.cos(Date.now() / 2000 + 20) / 100;
			} else if (key === 'floating_ring') {
				pco.rotation.x = Math.sin(Date.now() / 2000 + 50) / 100;
				pco.rotation.y = Math.cos(Date.now() / 2000 + 50) / 100;
				pco.rotation.z = Math.cos(Date.now() / 2000 + 50) / 100;
			} else if (key === 'jumbo') {
				pco.rotation.x = Math.sin(Date.now() / 5000 + 50) / 700;
				pco.rotation.y = Math.cos(Date.now() / 5000 + 50) / 700;
				pco.rotation.z = Math.cos(Date.now() / 5000 + 50) / 700;
			}
        });

        potree.updatePointClouds(Object.values(pointClouds), camera, renderer);
        renderer.render(scene, camera);
    }

    animate();

    document.body.onresize = function() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    };

    // @ts-ignore
    document.body.onresize();
};
