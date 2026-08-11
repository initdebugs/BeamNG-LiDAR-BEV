import QtQuick
import QtQuick.Controls
import QtQuick3D

Rectangle {
    id: root
    color: "#d7dadc"

    View3D {
        id: world
        anchors.fill: parent

        environment: SceneEnvironment {
            // Must stay equal to config.WORLD_AIR_RGB, which is what the depth
            // tint mixes toward. test_world_palette.py pins the two together.
            clearColor: "#d7dadc"
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High

            // There used to be a `fog: Fog { ... }` block here to dissolve
            // distant geometry into the air colour, carrying a note that
            // whether Qt applies fog to materials with lighting disabled was
            // undocumented. It was measured on a real GPU (Qt 6.7.1, D3D11)
            // and it does NOT: a red fog from 20 m to 60 m over geometry
            // spanning 5-75 m of NoLighting material did not move one channel.
            // Every large surface in this scene is NoLighting, so the block was
            // decorative and is gone. Range is now cued by the per-vertex tint
            // baked in world_scene.depth_tint, which no shader can ignore.
        }

        PerspectiveCamera {
            id: sceneCamera
            x: sceneBridge.cameraX
            y: sceneBridge.cameraY
            z: sceneBridge.cameraZ
            property real pitchAngle: sceneBridge.cameraPitch
            property real yawAngle: sceneBridge.cameraYaw
            eulerRotation: Qt.vector3d(pitchAngle, yawAngle, 0)
            fieldOfView: 48
            clipNear: 0.1
            // Must clear WORLD_RADIUS_M plus the trailing camera distance, or
            // the far field is clipped away exactly where the scene is trying
            // to show reach. 150 m of scene, up to 66 m of chase distance.
            clipFar: 420

            Behavior on x { NumberAnimation { duration: 240; easing.type: Easing.OutCubic } }
            Behavior on y { NumberAnimation { duration: 240; easing.type: Easing.OutCubic } }
            Behavior on z { NumberAnimation { duration: 240; easing.type: Easing.OutCubic } }
            Behavior on pitchAngle { NumberAnimation { duration: 240; easing.type: Easing.OutCubic } }
            Behavior on yawAngle { NumberAnimation { duration: 300; easing.type: Easing.OutCubic } }
        }
        camera: sceneCamera

        DirectionalLight {
            eulerRotation: Qt.vector3d(-46, -28, 0)
            brightness: 1.15
            castsShadow: true
            shadowFactor: 28
            shadowMapQuality: Light.ShadowMapQualityHigh
        }

        DirectionalLight {
            eulerRotation: Qt.vector3d(-25, 150, 0)
            brightness: 0.35
            castsShadow: false
        }

        // TWO luminance steps, then hue. Every ratio below is computed, not
        // estimated -- see tests/test_world_palette.py, which recomputes them.
        //
        // A light air caps what luminance alone can do: air is Y 0.698, so
        // air-to-black is 14.96:1 in total and only supports two steps at the
        // 3:1 floor for a graphical object (3^3 = 27 > 15). Trying to give
        // road, boundary, actors and the ego a rung each just moves the
        // failure around -- lifting the road to separate it from buildings
        // pushes the road into the air. So the ladder is:
        //
        //   air  -> road      3.53:1   the drivable surface against the void
        //   road -> obstacle  3.46:1   everything solid, in one dark band
        //
        // and inside that dark band things separate by HUE and by lighting:
        // boundary is unlit near-black neutral shaded by FACE, actors are lit
        // blue with real shading and a glass panel, the ego is neutral and
        // always centre screen. The old palette had this inverted -- empty air
        // was the BRIGHTEST surface, so a building silhouetted against it was
        // 1.35:1.
        //
        // The four unlit materials below carry NO colour of their own: they
        // ship white and multiply by a per-vertex colour. Their palette entries
        // live in config.py as WORLD_*_RGB. Two things have to vary WITHIN one
        // mesh and a material constant can express neither -- RANGE, so a wall
        // and the building behind it stop reading as one silhouette, and FACE
        // ORIENTATION, so an unlit box has edges at all. Measured on a real
        // GPU: a NoLighting DefaultMaterial multiplies its base by the vertex
        // colour in LINEAR space, so a white base renders srgb_to_linear(c) as
        // exactly c.
        DefaultMaterial {
            id: roadMaterial
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        DefaultMaterial {
            id: boundaryMaterial
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        DefaultMaterial {
            id: pathMaterial
            // The one element that deliberately breaks the dark-is-important
            // ramp: it is a guidance overlay rather than something perceived,
            // it is drawn over road and against air at the far end, and chroma
            // is its real channel. 1.94:1 vs road and 1.82:1 vs air. It is also
            // the one surface carrying no depth tint -- hazing the far end of
            // the path would fade out the part that says where the car is going
            // -- so its vertex colour is flat, and the alert swap moved into
            // world_scene._planned_path rather than being a binding here.
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        DefaultMaterial {
            id: routeMaterial
            // The navigation route: guidance like the path, subordinate to it
            // by chroma and alpha (world_scene._route_ribbon carries both in
            // the vertex colour), so the same flat white base.
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        DefaultMaterial {
            id: vehicleMaterial
            // Traffic as the LiDAR saw it, in the same blue as the corroborated
            // actor models. Hue is the only channel separating a car from a
            // wall -- both live in the one dark obstacle band -- so a car keeps
            // its colour whether or not the simulator ever confirms it.
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        // The emergency-braking overlay is two meshes: the corridor (rails,
        // wash, brake-now bar) and the threat marker (panel, frame, pool).
        //
        // Every element's transparency is PER-VERTEX, measured to multiply the
        // material opacity exactly, so these materials stay fully opaque and the
        // buffer carries the whole ramp. That is what lets one mesh hold a 0.04
        // wash and a 0.80 rail at once. Violet-while-armed and red-while-braking
        // likewise come from world_scene rather than from a binding here, which
        // keeps the drawn state provably the scanned state.
        DefaultMaterial {
            id: aebCorridorMaterial
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
            blendMode: DefaultMaterial.SourceOver
        }

        DefaultMaterial {
            id: aebMarkerMaterial
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
            blendMode: DefaultMaterial.SourceOver
            // The one animated thing in the scene, and only while the pedal is
            // actually down. A full-authority brake is the single event worth
            // taking the driver's eye, and motion does that where another static
            // shape would not. It rides on the material rather than on the
            // geometry so the pulse costs nothing per frame on the scene thread.
            opacity: 1.0
            SequentialAnimation on opacity {
                running: sceneBridge.alertText.length > 0
                loops: Animation.Infinite
                alwaysRunToEnd: true
                NumberAnimation {
                    to: 0.55
                    duration: 260
                    easing.type: Easing.InOutSine
                }
                NumberAnimation {
                    to: 1.0
                    duration: 260
                    easing.type: Easing.InOutSine
                }
                onStopped: aebMarkerMaterial.opacity = 1.0
            }
        }

        DefaultMaterial {
            id: uncertainMaterial
            // Deliberately the weakest mark on the road at 1.37:1. It is the
            // least confident thing drawn, so it should read that way.
            diffuseColor: "#ffffff"
            vertexColorsEnabled: true
            lighting: DefaultMaterial.NoLighting
            pointSize: 2.5
        }

        PrincipledMaterial {
            id: egoMaterial
            baseColor: "#1c2126"   // 3.27:1 vs road
            roughness: 0.62
            metalness: 0.08
        }

        PrincipledMaterial {
            id: glassMaterial
            // Lighter than the body now that the body is dark, so the
            // greenhouse still reads as a separate surface.
            baseColor: "#4a5f74"
            roughness: 0.28
            metalness: 0.12
        }

        PrincipledMaterial {
            id: actorMaterial
            // Blue, not neutral, and this is the point: traffic shares the
            // dark obstacle band with buildings, so hue is the only thing
            // telling a moving car from a static wall. The first attempt at
            // this palette put actors at #394046 against a #3c4348 boundary --
            // 1.05:1, indistinguishable. Now 1.51:1 apart AND hue-separated,
            // and lit, so it shades while the wall stays flat.
            baseColor: "#1b3c5c"
            roughness: 0.72
            metalness: 0.04
        }

        PrincipledMaterial {
            id: contactShadowMaterial
            baseColor: "#1b1f23"
            opacity: 0.09
            alphaMode: PrincipledMaterial.Blend
            lighting: PrincipledMaterial.NoLighting
        }

        // Geometry arrives as QQuick3DGeometry objects that own raw byte
        // buffers, not as ProceduralMesh value lists. ProceduralMesh takes
        // list<vector3d>, so feeding it cost one QVector3D built in a Python
        // loop per vertex per frame on the GUI thread -- the detail of the view
        // was limited by that loop rather than by what the sensors resolve.
        // These bind once and never rebind; new data arrives inside them.
        Model {
            geometry: sceneBridge.roadGeometry
            materials: [roadMaterial]
            castsShadows: false
            receivesShadows: true
        }

        Model {
            geometry: sceneBridge.boundaryGeometry
            materials: [boundaryMaterial]
            castsShadows: false
        }

        Model {
            geometry: sceneBridge.vehicleGeometry
            materials: [vehicleMaterial]
            castsShadows: false
        }

        // Before the path so the plan ribbon always reads over the route it
        // is following.
        Model {
            geometry: sceneBridge.routeGeometry
            materials: [routeMaterial]
            castsShadows: false
        }

        Model {
            geometry: sceneBridge.pathGeometry
            materials: [pathMaterial]
            castsShadows: false
        }

        // Drawn after the path so an emergency brake reads over the plan it is
        // overriding, which is the order the two actually resolve in.
        Model {
            geometry: sceneBridge.aebGeometry
            materials: [aebCorridorMaterial]
            castsShadows: false
        }

        Model {
            geometry: sceneBridge.aebMarkerGeometry
            materials: [aebMarkerMaterial]
            castsShadows: false
        }

        Model {
            geometry: sceneBridge.uncertainGeometry
            materials: [uncertainMaterial]
            castsShadows: false
        }

        Node {
            id: ego
            // Centred on the BODY, not on the render origin: the origin is the
            // vehicle's reference node, which sits off-centre in the bounding
            // box, and a car drawn at the origin stands beside where the
            // (correct) scene says it is.
            x: sceneBridge.egoCentreX
            y: -0.5
            z: sceneBridge.egoCentreZ

            // The ego casts a real shadow, but nothing can receive it: every
            // large surface is a NoLighting material, and those skip the
            // lighting path entirely, so `receivesShadows` on the road mesh is
            // a no-op. Stacking translucent discs fakes the contact shadow
            // instead -- five of them give a radial falloff, and without it the
            // car reads as floating above the road rather than sitting on it.
            Repeater3D {
                model: 5

                delegate: Model {
                    source: "#Cylinder"
                    y: 0.004 + index * 0.003
                    scale: Qt.vector3d(
                        sceneBridge.egoWidth * (1.5 - index * 0.16) / 100,
                        0.0002,
                        sceneBridge.egoLength * (1.16 - index * 0.12) / 100
                    )
                    materials: [contactShadowMaterial]
                    castsShadows: false
                    receivesShadows: false
                }
            }

            Model {
                source: "#Cube"
                y: sceneBridge.egoHeight * 0.34
                scale: Qt.vector3d(
                    sceneBridge.egoWidth / 100,
                    sceneBridge.egoHeight * 0.52 / 100,
                    sceneBridge.egoLength / 100
                )
                materials: [egoMaterial]
                castsShadows: true
            }

            Model {
                source: "#Cube"
                y: sceneBridge.egoHeight * 0.71
                z: sceneBridge.egoLength * 0.04
                scale: Qt.vector3d(
                    sceneBridge.egoWidth * 0.72 / 100,
                    sceneBridge.egoHeight * 0.34 / 100,
                    sceneBridge.egoLength * 0.48 / 100
                )
                materials: [glassMaterial]
                castsShadows: true
            }
        }

        Repeater3D {
            model: sceneBridge.actorModel

            delegate: Node {
                id: actorNode
                x: model.x
                y: model.y - 0.45
                z: model.z
                eulerRotation.y: model.yaw
                opacity: Math.max(0.12, model.confidence)

                Behavior on x { NumberAnimation { duration: 140; easing.type: Easing.OutQuad } }
                Behavior on y { NumberAnimation { duration: 140; easing.type: Easing.OutQuad } }
                Behavior on z { NumberAnimation { duration: 140; easing.type: Easing.OutQuad } }
                Behavior on opacity { NumberAnimation { duration: 180 } }
                Behavior on eulerRotation.y { NumberAnimation { duration: 140; easing.type: Easing.OutQuad } }

                Model {
                    source: "#Cube"
                    y: model.actorHeight * 0.34
                    scale: Qt.vector3d(
                        model.actorWidth / 100,
                        model.actorHeight * 0.52 / 100,
                        model.actorLength / 100
                    )
                    materials: [actorMaterial]
                    castsShadows: true
                }

                Model {
                    source: "#Cube"
                    y: model.actorHeight * 0.70
                    z: model.actorLength * 0.03
                    scale: Qt.vector3d(
                        model.actorWidth * 0.72 / 100,
                        model.actorHeight * 0.32 / 100,
                        model.actorLength * 0.48 / 100
                    )
                    materials: [glassMaterial]
                    castsShadows: true
                }
            }
        }
    }

    Rectangle {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 16
        width: 150
        height: 76
        radius: 11
        color: "#eaf7f8f9"
        border.color: "#f4ffffff"

        Text {
            x: 13
            y: 10
            text: "EGO SPEED"
            color: "#737a80"
            font.pixelSize: 10
            font.bold: true
            font.letterSpacing: 0.8
        }
        Text {
            x: 13
            y: 27
            text: sceneBridge.speedText
            color: "#25282b"
            font.pixelSize: 25
            font.weight: Font.DemiBold
        }
        Text {
            x: 56
            y: 39
            text: "km/h"
            color: "#747a80"
            font.pixelSize: 10
        }
        Text {
            x: 13
            y: 59
            text: sceneBridge.autonomyMode === "OFF"
                  ? "MANUAL"
                  : "SELF-DRIVING · " + sceneBridge.autonomyMode
            color: sceneBridge.autonomyMode === "OFF" ? "#737a80" : "#1a5fa5"
            font.pixelSize: 10
            font.bold: true
        }
    }

    Rectangle {
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: 16
        width: 140
        height: 62
        radius: 11
        color: "#eaf7f8f9"
        border.color: "#f4ffffff"

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            y: 9
            text: "PLANNED SPEED"
            color: "#737a80"
            font.pixelSize: 10
            font.bold: true
            font.letterSpacing: 0.7
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            y: 26
            text: sceneBridge.targetSpeedText + " km/h"
            color: "#25282b"
            font.pixelSize: 20
            font.weight: Font.DemiBold
        }
    }

    Rectangle {
        visible: sceneBridge.alertText.length > 0
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 17
        width: alertLabel.implicitWidth + 28
        height: 38
        radius: 9
        color: "#f2c0271e"
        border.color: "#ffd2cf"

        Text {
            id: alertLabel
            anchors.centerIn: parent
            text: sceneBridge.alertText
            color: "#ffffff"
            font.pixelSize: 12
            font.bold: true
        }
    }

    Rectangle {
        visible: !sceneBridge.perceptionAvailable
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 17
        width: perceptionLabel.implicitWidth + 30
        height: 34
        radius: 8
        color: "#c8353a40"

        Text {
            id: perceptionLabel
            anchors.centerIn: parent
            text: "PERCEPTION UNAVAILABLE"
            color: "#f0f2f4"
            font.pixelSize: 10
            font.bold: true
            font.letterSpacing: 0.8
        }
    }
}
