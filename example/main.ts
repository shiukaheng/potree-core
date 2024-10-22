import { AmbientLight, BoxGeometry, Euler, Group, Matrix4, Mesh, MeshBasicMaterial, PerspectiveCamera, Raycaster, Scene, SphereGeometry, Vector2, Vector3, WebGLRenderer } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';
import { PointCloudOctree, Potree } from '../source';

document.body.onload = function() {
    const potree = new Potree();
	potree.pointBudget = 500000
    let pointClouds: { [key: string]: PointCloudOctree } = {};

    // three.js
    const scene = new Scene();
    const camera = new PerspectiveCamera(60, 1, 0.1, 1000);

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
			scene.rotation.set(-Math.PI/2, 0, -Math.PI/2);
			scene.position.set(2, 0, 0);
        });
    }

    function addToUpdater(pco: PointCloudOctree): void {
        // Add to point clouds updater
    }

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

    function render() {
        // Update each point cloud independently if needed
        Object.keys(pointClouds).forEach(key => {
            const pco = pointClouds[key];
            // You can animate each pco object individually here
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
