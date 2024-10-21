import { AmbientLight, BoxGeometry, Euler, Group, Matrix4, Mesh, MeshBasicMaterial, PerspectiveCamera, Raycaster, Scene, SphereGeometry, Vector2, Vector3, WebGLRenderer } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';
import { PointCloudOctree, Potree } from '../source';

document.body.onload = function() {
    const potree = new Potree();
	potree.pointBudget = 500000
    let pointClouds: PointCloudOctree[] = [];

    // three.js
    const scene = new Scene();
    const camera = new PerspectiveCamera(60, 1, 0.1, 1000);

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
        powerPreference: 'high-performance'
    });

    // Enable XR
    renderer.xr.enabled = true;
    scene.add(new AmbientLight(0xffffff));
    camera.position.z = 0;
    loadPointCloud('/', 'metadata.json');

    function loadPointCloud(baseUrl: string, url: string, position?: Vector3, rotation?: Euler, scale?: Vector3) {
        potree.loadPointCloud(url, url => `${baseUrl}${url}`).then(function(pco: PointCloudOctree) {
            pco.material.size = 1.0;
            pco.material.shape = 0;
            pco.material.inputColorEncoding = 1;
            pco.material.outputColorEncoding = 1;

            if (position) { pco.position.copy(position); }
            if (rotation) { pco.rotation.copy(rotation); }
            if (scale) { pco.scale.copy(scale); }

            console.log('Pointcloud file loaded', pco);
            pco.showBoundingBox = false;

            const box = pco.pcoGeometry.boundingBox;
            const size = box.getSize(new Vector3());

            const geometry = new BoxGeometry(size.x, size.y, size.z);
            const material = new MeshBasicMaterial({color: 0xFF0000, wireframe: true});
            const mesh = new Mesh(geometry, material);
            mesh.position.copy(pco.position);
            mesh.scale.copy(pco.scale);
            mesh.rotation.copy(pco.rotation);
            mesh.raycast = () => false;

            size.multiplyScalar(0.5);
            mesh.position.add(new Vector3(size.x, size.y, -size.z));

            scene.add(mesh);

            addToUpdater(pco);

			const group = new Group();
			group.add(pco);

			scene.add(group);
			scene.rotation.set(-Math.PI/2, 0, -Math.PI/2);
			scene.position.set(0, 3, 0)
        });
    }

    function addToUpdater(pco: PointCloudOctree): void {
        // scene.add(pco);
        pointClouds.push(pco);
    }

    function unload(): void {
        pointClouds.forEach(pco => {
            scene.remove(pco);
            pco.dispose();
        });

        pointClouds = [];
    }

    // WebXR setup
    document.body.appendChild(VRButton.createButton(renderer));

    function animate() {
        renderer.setAnimationLoop(render);
    }

    function render() {
        potree.updatePointClouds(pointClouds, camera, renderer);
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