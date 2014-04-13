#include "window.h"

window::window(QWidget *parent) :
    QWidget(parent)
{
    Render = new render;
}


#include "renderarea.h"
#include "window.h"

#include <QtWidgets>

//! [0]
const int IdRole = Qt::UserRole;
//! [0]

//! [1]
Window::Window()
{
    renderArea = new RenderArea;
