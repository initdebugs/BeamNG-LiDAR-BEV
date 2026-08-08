import QtQuick
import QtQuick.Controls
import QtQuick3D
import QtQuick3D.Helpers

Rectangle {
    id: root
    color: "#d7dadc"

    View3D {
        id: world
        anchors.fill: parent

        environment: SceneEnvironment {
            clearColor: "#d7dadc"
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
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
            clipFar: 260

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

        DefaultMaterial {
            id: roadMaterial
            diffuseColor: "#808689"
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        DefaultMaterial {
            id: boundaryMaterial
            diffuseColor: "#b8bdc0"
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        DefaultMaterial {
            id: pathMaterial
            diffuseColor: sceneBridge.alertText.length > 0
                          ? "#eb423c"
                          : "#297bd3"
            lighting: DefaultMaterial.NoLighting
            cullMode: DefaultMaterial.NoCulling
        }

        DefaultMaterial {
            id: uncertainMaterial
            diffuseColor: "#82909a"
            lighting: DefaultMaterial.NoLighting
            pointSize: 2.5
        }

        PrincipledMaterial {
            id: egoMaterial
            baseColor: "#f3f4f5"
            roughness: 0.62
            metalness: 0.08
        }

        PrincipledMaterial {
            id: glassMaterial
            baseColor: "#4f5a61"
            roughness: 0.28
            metalness: 0.12
        }

        PrincipledMaterial {
            id: actorMaterial
            baseColor: "#8f9599"
            roughness: 0.72
            metalness: 0.04
        }

        Model {
            geometry: ProceduralMesh {
                positions: sceneBridge.roadPositions
                indexes: sceneBridge.roadIndices
            }
            materials: [roadMaterial]
            castsShadows: false
            receivesShadows: true
        }

        Model {
            geometry: ProceduralMesh {
                positions: sceneBridge.boundaryPositions
                indexes: sceneBridge.boundaryIndices
            }
            materials: [boundaryMaterial]
            castsShadows: false
        }

        Model {
            geometry: ProceduralMesh {
                positions: sceneBridge.pathPositions
                indexes: sceneBridge.pathIndices
            }
            materials: [pathMaterial]
            castsShadows: false
        }

        Model {
            geometry: ProceduralMesh {
                positions: sceneBridge.uncertainPositions
                primitiveMode: ProceduralMesh.Points
            }
            materials: [uncertainMaterial]
            castsShadows: false
        }

        Node {
            id: ego
            y: -0.5

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
            color: sceneBridge.autonomyMode === "OFF" ? "#737a80" : "#337bc9"
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
        color: "#e8bb342f"
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
