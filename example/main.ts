import { AmbientLight, BoxGeometry, Euler, Matrix4, Mesh, MeshBasicMaterial, PerspectiveCamera, Raycaster, Scene, SphereGeometry, Vector2, Vector3, WebGLRenderer } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';
import { PointCloudOctree, Potree } from '../source';

document.body.onload = function() {
    const potree = new Potree();
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

    // const controls = new OrbitControls(camera, canvas);
    camera.position.z = 0;

    const raycaster = new Raycaster();
    // @ts-ignore
    raycaster.params.Points.threshold = 1e-2;
    const normalized = new Vector2();

    canvas.onmousemove = function(event) {
        normalized.set(event.clientX / canvas.width * 2 - 1, -(event.clientY / canvas.height) * 2 + 1);
        raycaster.setFromCamera(normalized, camera);
    };

    canvas.ondblclick = function() {
        const intersects = raycaster.intersectObject(scene, true);

        if (intersects.length > 0) {
            const geometry = new SphereGeometry(0.2, 32, 32);
            const material = new MeshBasicMaterial({color: Math.random() * 0xAA4444});
            const sphere = new Mesh(geometry, material);
            sphere.position.copy(intersects[0].point);
            scene.add(sphere);
        }
    };

    // loadPointCloud('/data/lion_takanawa/', 'cloud.js', new Vector3(-4, -2, 5), new Euler(-Math.PI / 2, 0, 0));
    loadPointCloud('/', 'metadata.json', new Vector3(-21, -1, -6), new Euler(-Math.PI / 2, 0, -Math.PI / 2), new Vector3(1, 1, 1));

    function loadPointCloud(baseUrl: string, url: string, position?: Vector3, rotation?: Euler, scale?: Vector3) {
        potree.loadPointCloud(url, url => `${baseUrl}${url}`).then(function(pco: PointCloudOctree) {
            pco.material.size = 1.0;
            pco.material.shape = 2;
            pco.material.inputColorEncoding = 1;
            pco.material.outputColorEncoding = 1;

            if (position) { pco.position.copy(position); }
            if (rotation) { pco.rotation.copy(rotation); }
            // if (scale) { pco.scale.copy(scale); }

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

            add(pco);
        });
    }

    function add(pco: PointCloudOctree): void {
        scene.add(pco);
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

    // VR Controllers
    const controllerModelFactory = new XRControllerModelFactory();

    const controller1 = renderer.xr.getController(0);
    scene.add(controller1);

    const controllerGrip1 = renderer.xr.getControllerGrip(0);
    controllerGrip1.add(controllerModelFactory.createControllerModel(controllerGrip1));
    scene.add(controllerGrip1);

    const controller2 = renderer.xr.getController(1);
    scene.add(controller2);

    const controllerGrip2 = renderer.xr.getControllerGrip(1);
    controllerGrip2.add(controllerModelFactory.createControllerModel(controllerGrip2));
    scene.add(controllerGrip2);

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